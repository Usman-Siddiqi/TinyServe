"""Phase 1: the model and its external KV cache."""

import torch

from tinyserve.kv_cache import ContiguousKVCache, KVPool
from tinyserve.model import ModelConfig, TinyGPT
from tinyserve.tokenizer import CharTokenizer

CFG = ModelConfig(max_seq_len=64)
ATOL = 1e-5


def make_cache(cfg: ModelConfig, rows: int) -> ContiguousKVCache:
    pool = KVPool(rows * cfg.max_seq_len, cfg.n_layers, cfg.n_heads, cfg.head_dim)
    return ContiguousKVCache(pool, cfg.max_seq_len)


def random_ids(n: int, seed: int = 0, batch: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, CFG.vocab_size, (batch, n), generator=g)


def test_output_shape() -> None:
    model = TinyGPT(CFG)
    ids = random_ids(10, batch=3)
    assert model(ids).shape == (3, 10, CFG.vocab_size)


def test_same_seed_same_weights_and_outputs() -> None:
    a, b = TinyGPT(CFG), TinyGPT(CFG)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)
    ids = random_ids(12)
    assert torch.equal(a(ids), b(ids))


def test_different_seed_different_weights() -> None:
    a, b = TinyGPT(CFG), TinyGPT(ModelConfig(max_seq_len=64, seed=1))
    assert not torch.equal(a.tok_emb.weight, b.tok_emb.weight)


def test_model_init_does_not_touch_global_rng() -> None:
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    TinyGPT(CFG)
    assert torch.equal(torch.rand(3), expected)


@torch.inference_mode()
def test_incremental_decoding_matches_full_recompute() -> None:
    """Prefill a prompt into the cache, then feed one token at a time.

    The logits for every position must match one big forward pass without a cache.
    """
    model = TinyGPT(CFG)
    ids = random_ids(20, seed=1)
    full = model(ids)

    cache = make_cache(CFG, rows=1)
    cache.allocate(seq_id=0, num_tokens=8)
    prompt, rest = ids[:, :8], ids[:, 8:]
    positions = torch.arange(8)[None]
    kv = cache.build_batch([0], positions, torch.ones_like(prompt, dtype=torch.bool))
    step_logits = [model(prompt, positions, kv)]
    for j in range(rest.shape[1]):
        pos = torch.tensor([[8 + j]])
        cache.grow(0, 9 + j)
        kv = cache.build_batch([0], pos, torch.ones(1, 1, dtype=torch.bool))
        step_logits.append(model(rest[:, j : j + 1], pos, kv))

    torch.testing.assert_close(torch.cat(step_logits, dim=1), full, atol=ATOL, rtol=0)


@torch.inference_mode()
def test_batched_ragged_decode_matches_individual() -> None:
    """Sequences of different lengths in one padded batch give the same logits as alone."""
    model = TinyGPT(CFG)
    lengths = [5, 11, 3]
    seqs = [random_ids(n + 1, seed=i)[0] for i, n in enumerate(lengths)]
    cache = make_cache(CFG, rows=len(lengths))

    # Prefill all prompts in one right-padded batch.
    max_len = max(lengths)
    ids = torch.zeros(len(lengths), max_len, dtype=torch.long)
    valid = torch.zeros(len(lengths), max_len, dtype=torch.bool)
    for b, (seq, n) in enumerate(zip(seqs, lengths, strict=True)):
        ids[b, :n] = seq[:n]
        valid[b, :n] = True
        cache.allocate(b, n + 1)
    positions = torch.arange(max_len).expand(len(lengths), -1) * valid
    model(ids, positions, cache.build_batch([0, 1, 2], positions, valid))

    # One decode step for everyone at their own position.
    next_ids = torch.stack([seq[n] for seq, n in zip(seqs, lengths, strict=True)])[:, None]
    pos = torch.tensor(lengths)[:, None]
    kv = cache.build_batch([0, 1, 2], pos, torch.ones_like(pos, dtype=torch.bool))
    batched = model(next_ids, pos, kv)[:, 0]

    for b, seq in enumerate(seqs):
        alone = model(seq[None])[0, -1]
        torch.testing.assert_close(batched[b], alone, atol=ATOL, rtol=0)


def test_tokenizer_roundtrip() -> None:
    tok = CharTokenizer()
    text = "Hello, TinyServe!\n"
    ids = tok.encode(text)
    assert ids[0] == tok.bos_id
    assert tok.decode(ids) == text
    assert max(ids) < tok.vocab_size
    assert tok.encode("é", add_bos=False) == [tok.unk_id]
