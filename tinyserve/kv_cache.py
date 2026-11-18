"""KV cache memory: one physical pool, and managers that decide who owns which slots.

During decoding, every new token attends to the keys and values of all previous
tokens in its sequence. Recomputing them each step would make generation
quadratic, so we cache them. The cache is by far the largest piece of serving
memory, and *how it is laid out* decides how many requests fit at once.

This module splits the problem in two:

* :class:`KVPool` is dumb storage: per layer, one flat tensor of ``num_slots``
  token slots, each holding a key and a value of shape ``(n_kv_heads, head_dim)``,
  in float32, bfloat16 or int8 (see :mod:`tinyserve.kv_quant`).
* A :class:`KVCacheManager` owns the bookkeeping: which slots belong to which
  sequence. It turns "token ``j`` of sequence ``s``" into a physical slot address.

Both layouts use the same address function, ``block_table[j // B] * B + j % B``,
and differ only in their block size ``B`` and in who gets which block:

* :class:`PagedKVCache` hands out small fixed-size blocks on demand, like the
  pages of virtual memory (and vLLM's PagedAttention).
* :class:`ContiguousKVCache` gives each sequence one slab of ``max_seq_len``
  slots up front: a single block with ``B = max_seq_len``, so the address is
  simply ``row * max_seq_len + j``.

:class:`PrefixCachingKVCache` adds automatic prefix caching on top of paging:
full blocks are shared between sequences that start with the same tokens.
"""

from __future__ import annotations

import hashlib
import itertools
from abc import ABC, abstractmethod
from array import array
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from tinyserve.kv_quant import dequantize_int8, kv_format, quantize_int8


class KVPool:
    """Physical KV storage: ``k[layer]`` and ``v[layer]`` of shape ``(num_slots, H, D)``.

    ``kv_dtype`` picks the storage format (see :mod:`tinyserve.kv_quant`). For
    ``"int8"`` each (token, head) vector also has a float32 scale in ``k_scale`` /
    ``v_scale``. Reads return keys/values in the format's compute dtype.
    """

    def __init__(
        self,
        num_slots: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        kv_dtype: str = "float32",
    ) -> None:
        self.num_slots = num_slots
        self.format = kv_format(kv_dtype)
        shape = (n_layers, num_slots, n_kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=self.format.storage)
        self.v = torch.zeros(shape, dtype=self.format.storage)
        if self.format.quantized:
            self.k_scale = torch.zeros(shape[:-1])
            self.v_scale = torch.zeros(shape[:-1])
        # Optional per-channel key divisors (n_layers, H, D); see calibrate_key_smoothing.
        self.key_smoothing: Tensor | None = None

    @staticmethod
    def bytes_per_token(n_layers: int, n_kv_heads: int, head_dim: int, kv_dtype: str) -> int:
        """Bytes one token slot takes across all layers (keys + values, scales included)."""
        return 2 * n_layers * n_kv_heads * kv_format(kv_dtype).bytes_per_head(head_dim)

    @property
    def kv_dtype(self) -> str:
        """Name of the storage format."""
        return self.format.name

    @property
    def nbytes(self) -> int:
        """Total bytes held by the pool (keys + values + scales, all layers)."""
        total = self.k.nbytes + self.v.nbytes
        if self.format.quantized:
            total += self.k_scale.nbytes + self.v_scale.nbytes
        return total

    def write(self, layer: int, slots: Tensor, k: Tensor, v: Tensor) -> None:
        """Store ``(N, H, D)`` keys/values at ``slots`` ``(N,)``."""
        if self.key_smoothing is not None:
            k = k / self.key_smoothing[layer]
        if self.format.quantized:
            k, k_scale = quantize_int8(k)
            v, v_scale = quantize_int8(v)
            self.k_scale[layer].index_copy_(0, slots, k_scale)
            self.v_scale[layer].index_copy_(0, slots, v_scale)
        # index_copy_/index_select are several times faster than advanced indexing on CPU.
        self.k[layer].index_copy_(0, slots, k.to(self.format.storage))
        self.v[layer].index_copy_(0, slots, v.to(self.format.storage))

    def smooth_queries(self, layer: int, q: Tensor) -> Tensor:
        """Scale ``(..., n_heads, D)`` queries by the key smoothing so ``q . k`` is unchanged."""
        if self.key_smoothing is None:
            return q
        s = self.key_smoothing[layer]
        return q * s.repeat_interleave(q.shape[-2] // s.shape[0], dim=0)

    def read(self, layer: int, slots: Tensor) -> tuple[Tensor, Tensor]:
        """Gather ``slots`` ``(G, L)`` into dense ``(G, L, H, D)`` tensors of the compute dtype.

        With key smoothing on, keys come back divided by it (see :meth:`smooth_queries`).
        """
        flat = slots.reshape(-1)
        shape = (*slots.shape, *self.k.shape[2:])
        k = self.k[layer].index_select(0, flat).view(shape)
        v = self.v[layer].index_select(0, flat).view(shape)
        if self.format.quantized:
            dtype = self.format.compute
            k_scale = self.k_scale[layer].index_select(0, flat).view(shape[:-1])
            v_scale = self.v_scale[layer].index_select(0, flat).view(shape[:-1])
            return dequantize_int8(k, k_scale, dtype), dequantize_int8(v, v_scale, dtype)
        return k, v


@dataclass
class AttentionGroup:
    """A subset of a batch's rows that attend together over a common padded length.

    Attributes:
        rows: ``(G,)`` indices of the batch rows in this group.
        read_slots: ``(G, L)`` slot address of token ``j`` of each row's sequence,
            for ``j < L``, where ``L`` is the longest context in the group.
            Entries beyond a sequence's length point at harmless garbage; the
            causal mask hides them.
        mask: ``(G, 1, S, L)`` bool, True where query ``i`` may see key ``j``.
            Built once per step and shared by every layer.
    """

    rows: Tensor
    read_slots: Tensor
    mask: Tensor


@dataclass
class KVBatch:
    """Per-forward-pass instructions telling attention where to write and read KV.

    Attributes:
        pool: the physical storage.
        write_rows: ``(N,)`` indices of the real (non-padding) new tokens in the
            flattened ``(B * S)`` batch.
        write_slots: ``(N,)`` slot address for each entry of ``write_rows``.
        groups: rows bucketed by context length. Attention is computed per group,
            so a 20-token sequence is not padded out to the 450-token one that
            happens to share its batch (see :meth:`KVCacheManager.build_batch`).
    """

    pool: KVPool
    write_rows: Tensor
    write_slots: Tensor
    groups: list[AttentionGroup]

    def write(self, layer: int, k: Tensor, v: Tensor) -> None:
        """Store the new ``(B, S, H, D)`` keys/values at their slots."""
        rows = self.write_rows
        self.pool.write(
            layer,
            self.write_slots,
            k.flatten(0, 1).index_select(0, rows),
            v.flatten(0, 1).index_select(0, rows),
        )

    def read(self, layer: int, read_slots: Tensor) -> tuple[Tensor, Tensor]:
        """Gather keys/values at ``read_slots`` ``(G, L)`` into dense ``(G, L, H, D)`` tensors."""
        return self.pool.read(layer, read_slots)


@dataclass(frozen=True)
class KVUsage:
    """A snapshot of KV memory usage, in tokens.

    Attributes:
        used_tokens: slots holding a real token's KV right now.
        reserved_tokens: slots handed out to sequences (used + reserved-but-empty).
        capacity_tokens: total slots in the pool.
        cached_tokens: free slots that still hold reusable KV (prefix caching only).
    """

    used_tokens: int
    reserved_tokens: int
    capacity_tokens: int
    cached_tokens: int = 0

    @property
    def utilization(self) -> float:
        """Fraction of *reserved* memory that holds real data (1.0 = no waste)."""
        return self.used_tokens / self.reserved_tokens if self.reserved_tokens else 1.0

    @property
    def occupancy(self) -> float:
        """Fraction of the pool that has been handed out to sequences."""
        return self.reserved_tokens / self.capacity_tokens if self.capacity_tokens else 0.0

    @property
    def wasted_tokens(self) -> int:
        """Reserved slots that hold nothing (internal fragmentation)."""
        return self.reserved_tokens - self.used_tokens


@dataclass
class PrefixCacheStats:
    """Cumulative prefix-cache counters, over every admission (re-admissions included).

    Attributes:
        query_tokens: tokens of the sequences admitted.
        hit_tokens: of those, tokens whose KV was already in the cache.
        evicted_blocks: cached blocks reclaimed to make room for new ones.
    """

    query_tokens: int = 0
    hit_tokens: int = 0
    evicted_blocks: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of admitted tokens that did not need to be prefilled."""
        return self.hit_tokens / self.query_tokens if self.query_tokens else 0.0


class KVCacheManager(ABC):
    """Bookkeeping for which pool slots belong to which sequence.

    The engine calls :meth:`allocate` when a sequence is admitted, :meth:`grow`
    before each step that adds tokens, and :meth:`free` when it finishes or is
    preempted. :meth:`build_batch` turns a batch of sequences and positions into
    the :class:`KVBatch` the model consumes. Subclasses choose a ``block_size``
    and implement :meth:`block_table`.
    """

    layout = "base"
    prefix_caching = False

    def __init__(self, pool: KVPool, block_size: int) -> None:
        self.pool = pool
        self.block_size = block_size
        self._lengths: dict[int, int] = {}

    # --- allocation -------------------------------------------------------------

    @abstractmethod
    def can_allocate(self, num_tokens: int, token_ids: Sequence[int] | None = None) -> bool:
        """Whether a *new* sequence needing ``num_tokens`` slots would fit right now."""

    def allocate(
        self, seq_id: int, num_tokens: int, token_ids: Sequence[int] | None = None
    ) -> bool:
        """Reserve memory for a new sequence of ``num_tokens``. Returns False if it won't fit.

        Given the sequence's ``token_ids``, a prefix-caching manager shares the
        blocks of its first :meth:`match_prefix` tokens instead of allocating them.
        """
        if seq_id in self._lengths:
            raise ValueError(f"sequence {seq_id} is already allocated")
        if not self.can_allocate(num_tokens, token_ids):
            return False
        if not self._allocate(seq_id, num_tokens, token_ids):
            return False
        self._lengths[seq_id] = num_tokens
        return True

    def grow(self, seq_id: int, num_tokens: int) -> bool:
        """Make sure ``seq_id`` has room for ``num_tokens`` in total. False if out of memory."""
        if num_tokens <= self._lengths[seq_id]:
            return True
        if not self._grow(seq_id, num_tokens):
            return False
        self._lengths[seq_id] = num_tokens
        return True

    def free(self, seq_id: int) -> None:
        """Release everything held by ``seq_id``."""
        self._free(seq_id)
        del self._lengths[seq_id]

    def __contains__(self, seq_id: int) -> bool:
        return seq_id in self._lengths

    @property
    def num_sequences(self) -> int:
        """Number of sequences currently holding memory."""
        return len(self._lengths)

    def match_prefix(self, token_ids: Sequence[int]) -> int:
        """How many leading tokens of ``token_ids`` already have their KV cached."""
        return 0

    def cache_blocks(self, seq_id: int, token_ids: Sequence[int]) -> None:  # noqa: B027
        """Record that the KV of ``token_ids``, a prefix of ``seq_id``, is computed."""

    def prefix_cache_stats(self) -> PrefixCacheStats | None:
        """Hit-rate counters, or None if this manager does not cache prefixes."""
        return None

    @abstractmethod
    def _allocate(self, seq_id: int, num_tokens: int, token_ids: Sequence[int] | None) -> bool: ...

    @abstractmethod
    def _grow(self, seq_id: int, num_tokens: int) -> bool: ...

    @abstractmethod
    def _free(self, seq_id: int) -> None: ...

    # --- addressing -------------------------------------------------------------

    @abstractmethod
    def block_table(self, seq_ids: list[int]) -> Tensor:
        """``(B, W)`` physical block ids of each sequence, right-padded with zeros."""

    def addresses(self, seq_ids: list[int], positions: Tensor) -> Tensor:
        """Map ``positions[b, j]`` of sequence ``seq_ids[b]`` to physical slot indices."""
        return self._slots(self.block_table(seq_ids), positions)

    def _slots(self, table: Tensor, positions: Tensor) -> Tensor:
        """``table[b, pos // block_size] * block_size + pos % block_size``."""
        logical_block = (positions // self.block_size).clamp(max=table.shape[1] - 1)
        return table.gather(1, logical_block) * self.block_size + positions % self.block_size

    def build_batch(
        self, seq_ids: list[int], positions: Tensor, valid: Tensor, group_size: int = 8
    ) -> KVBatch:
        """Build the :class:`KVBatch` for one forward pass.

        Attention has to gather each sequence's history into a dense tensor, and a
        dense batch is padded to its longest member. In a busy decode batch,
        lengths range from a few tokens to hundreds, so padding everyone to the
        maximum can double the memory traffic. Instead, rows are sorted by context
        length and attended in groups of ``group_size``, each padded only to its
        own longest member. (Real engines use variable-length attention kernels
        that need no padding at all; grouping gets most of the benefit in plain
        PyTorch.)

        Args:
            seq_ids: the sequences in the batch, one per row.
            positions: ``(B, S)`` absolute positions of the new tokens.
            valid: ``(B, S)`` bool, False for padding.
            group_size: rows per attention group.
        """
        table = self.block_table(seq_ids)  # built once; groups slice their rows from it
        write_slots = self._slots(table, positions)[valid]
        write_rows = valid.flatten().nonzero().squeeze(1)
        context_lens = (positions * valid).amax(dim=1) + 1
        order = torch.argsort(context_lens, descending=True, stable=True)
        groups = []
        for rows in order.split(group_size):
            key_positions = torch.arange(int(context_lens[rows[0]]))
            read_slots = self._slots(table[rows], key_positions.expand(len(rows), -1))
            # Causal by absolute position, which also hides padding and stale slots.
            mask = key_positions <= positions[rows][:, :, None]  # (G, S, L)
            groups.append(AttentionGroup(rows, read_slots, mask[:, None]))
        return KVBatch(self.pool, write_rows, write_slots, groups)

    # --- accounting -------------------------------------------------------------

    @abstractmethod
    def reserved_tokens(self) -> int:
        """Slots currently handed out to sequences."""

    def usage(self) -> KVUsage:
        """Current memory usage snapshot."""
        return KVUsage(
            used_tokens=sum(self._lengths.values()),
            reserved_tokens=self.reserved_tokens(),
            capacity_tokens=self.pool.num_slots,
        )


class ContiguousKVCache(KVCacheManager):
    """The classic layout: every sequence gets a private slab of ``max_seq_len`` slots.

    This is what a naive implementation (or Hugging Face's static cache) does. It
    is simple and the memory of each sequence is contiguous, but each slab is
    sized for the *longest possible* sequence. A request that ends up using 80
    tokens still pins 512 slots for its whole lifetime, which caps how many
    requests can share the pool.
    """

    layout = "contiguous"

    def __init__(self, pool: KVPool, max_seq_len: int) -> None:
        super().__init__(pool, block_size=max_seq_len)
        self.max_seq_len = max_seq_len
        self.num_rows = pool.num_slots // max_seq_len
        if self.num_rows == 0:
            raise ValueError("pool is smaller than a single max_seq_len slab")
        self._free_rows = list(range(self.num_rows - 1, -1, -1))  # pop() yields row 0 first
        self._row_of: dict[int, int] = {}

    def can_allocate(self, num_tokens: int, token_ids: Sequence[int] | None = None) -> bool:
        """A new sequence fits if any slab is free and it is not longer than a slab."""
        return bool(self._free_rows) and num_tokens <= self.max_seq_len

    def _allocate(self, seq_id: int, num_tokens: int, token_ids: Sequence[int] | None) -> bool:
        self._row_of[seq_id] = self._free_rows.pop()
        return True

    def _grow(self, seq_id: int, num_tokens: int) -> bool:
        return num_tokens <= self.max_seq_len

    def _free(self, seq_id: int) -> None:
        self._free_rows.append(self._row_of.pop(seq_id))

    def row_of(self, seq_id: int) -> int:
        """The slab index owned by ``seq_id``."""
        return self._row_of[seq_id]

    def block_table(self, seq_ids: list[int]) -> Tensor:
        """A single block per sequence: its slab."""
        return torch.tensor([[self._row_of[s]] for s in seq_ids], dtype=torch.long)

    def reserved_tokens(self) -> int:
        """Each live sequence pins a full slab."""
        return len(self._row_of) * self.max_seq_len


class OutOfBlocksError(RuntimeError):
    """Raised when the block allocator has no free blocks left."""


class BlockAllocator:
    """Hands out fixed-size physical blocks from a free list.

    This is the whole "memory allocator" of a paged KV cache: since every block
    has the same size, there is no external fragmentation and no compaction.
    Any free block is as good as any other.
    """

    def __init__(self, num_blocks: int) -> None:
        if num_blocks < 1:
            raise ValueError("need at least one block")
        self.num_blocks = num_blocks
        # A stack: the most recently freed block is reused first (it is warm in cache).
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))
        self._is_free = [True] * num_blocks

    @property
    def num_free(self) -> int:
        """Blocks currently available."""
        return len(self._free)

    @property
    def num_used(self) -> int:
        """Blocks currently handed out."""
        return self.num_blocks - self.num_free

    def allocate(self) -> int:
        """Take one block off the free list."""
        if not self._free:
            raise OutOfBlocksError("no free KV blocks")
        block = self._free.pop()
        self._is_free[block] = False
        return block

    def free(self, block: int) -> None:
        """Return a block to the free list."""
        if self._is_free[block]:
            raise ValueError(f"block {block} is already free (double free)")
        self._is_free[block] = True
        self._free.append(block)

    def free_blocks(self) -> Iterator[int]:
        """Iterate over the currently free block ids."""
        return iter(self._free)


BlockHash = bytes


def block_hash(parent: BlockHash | None, token_ids: Sequence[int]) -> BlockHash:
    """SHA-256 of a full block chained over its parent's, so it identifies the whole prefix.

    A cryptographic hash makes an accidental collision (which would silently reuse
    the wrong KV) practically impossible.
    """
    h = hashlib.sha256(parent or b"")
    h.update(array("q", token_ids).tobytes())
    return h.digest()


class CachingBlockAllocator(BlockAllocator):
    """A block allocator with reference counts and a content-addressed block cache.

    A block registered under a hash keeps its KV when its last owner frees it: it
    becomes *evictable* instead of free, and a later sequence with the same prefix
    can take it back with :meth:`acquire`. Evictable blocks count as free space and
    are reclaimed least-recently-freed first, once no clean block is left.
    """

    def __init__(self, num_blocks: int) -> None:
        super().__init__(num_blocks)
        self.ref_counts = [0] * num_blocks
        self.num_shared_refs = 0  # sum over blocks of (owners - 1)
        self.num_evictions = 0
        self._evictable: OrderedDict[int, None] = OrderedDict()  # least recently freed first
        self._hash_of: dict[int, BlockHash] = {}
        self._block_of: dict[BlockHash, int] = {}

    @property
    def num_free(self) -> int:
        """Blocks available: clean ones plus evictable cached ones."""
        return len(self._free) + len(self._evictable)

    @property
    def num_evictable(self) -> int:
        """Free blocks that still hold cached KV."""
        return len(self._evictable)

    def allocate(self) -> int:
        """Take a clean block, or evict the least recently freed cached block."""
        if self._free:
            block = self._free.pop()
        elif self._evictable:
            block, _ = self._evictable.popitem(last=False)
            del self._block_of[self._hash_of.pop(block)]
            self.num_evictions += 1
        else:
            raise OutOfBlocksError("no free KV blocks")
        self.ref_counts[block] = 1
        return block

    def free(self, block: int) -> None:
        """Drop one owner. An unowned block becomes evictable if cached, else clean."""
        refs = self.ref_counts[block]
        if refs == 0:
            raise ValueError(f"block {block} is already free (double free)")
        self.ref_counts[block] = refs - 1
        if refs > 1:
            self.num_shared_refs -= 1
        elif block in self._hash_of:
            self._evictable[block] = None
        else:
            self._free.append(block)

    def free_blocks(self) -> Iterator[int]:
        """Iterate over the free block ids, clean ones first."""
        return itertools.chain(self._free, self._evictable)

    def lookup(self, block_hash: BlockHash) -> int | None:
        """The cached block holding the content identified by ``block_hash``, if any."""
        return self._block_of.get(block_hash)

    def acquire(self, block: int) -> None:
        """Add an owner to a cached block, taking it off the evictable list if needed."""
        if self.ref_counts[block] == 0:
            del self._evictable[block]
        else:
            self.num_shared_refs += 1
        self.ref_counts[block] += 1

    def register(self, block: int, block_hash: BlockHash) -> None:
        """Make ``block`` findable by ``block_hash``. The first block with a hash wins."""
        if block_hash not in self._block_of and block not in self._hash_of:
            self._hash_of[block] = block_hash
            self._block_of[block_hash] = block


class PagedKVCache(KVCacheManager):
    """A paged KV cache: sequences own lists of small blocks, allocated on demand.

    Each sequence has a *block table*: logical block ``i`` (tokens
    ``i*B .. i*B+B-1``) lives in physical block ``block_table[i]``. Physical
    blocks of one sequence can be anywhere in the pool. A sequence only ever
    wastes the unused tail of its last block, so nearly all reserved memory holds
    real data, and many more sequences fit in the same pool.

    Args:
        pool: the physical storage; its slots are split into blocks of ``block_size``.
        block_size: tokens per block.
        watermark: fraction of blocks kept free when admitting *new* sequences, so
            running sequences have room to grow before anyone must be preempted.
    """

    layout = "paged"

    def __init__(self, pool: KVPool, block_size: int = 16, watermark: float = 0.01) -> None:
        super().__init__(pool, block_size)
        self.allocator = BlockAllocator(pool.num_slots // block_size)
        self.watermark_blocks = int(watermark * self.allocator.num_blocks)
        self.block_tables: dict[int, list[int]] = {}

    def blocks_needed(self, num_tokens: int) -> int:
        """Number of blocks that hold ``num_tokens`` tokens."""
        return -(-num_tokens // self.block_size)  # ceil division

    def can_allocate(self, num_tokens: int, token_ids: Sequence[int] | None = None) -> bool:
        """A new sequence fits if its blocks are free without dipping into the watermark."""
        needed = self.blocks_needed(num_tokens)
        return needed <= self.allocator.num_free - self.watermark_blocks

    def _allocate(self, seq_id: int, num_tokens: int, token_ids: Sequence[int] | None) -> bool:
        needed = self.blocks_needed(num_tokens)
        self.block_tables[seq_id] = [self.allocator.allocate() for _ in range(needed)]
        return True

    def _grow(self, seq_id: int, num_tokens: int) -> bool:
        table = self.block_tables[seq_id]
        extra = self.blocks_needed(num_tokens) - len(table)
        if extra > self.allocator.num_free:
            return False
        table.extend(self.allocator.allocate() for _ in range(extra))
        return True

    def _free(self, seq_id: int) -> None:
        for block in self.block_tables.pop(seq_id):
            self.allocator.free(block)

    def block_table(self, seq_ids: list[int]) -> Tensor:
        """Each sequence's block list, padded to the longest one."""
        tables = [self.block_tables[s] for s in seq_ids]
        width = max(len(t) for t in tables)
        return torch.tensor([t + [0] * (width - len(t)) for t in tables], dtype=torch.long)

    def reserved_tokens(self) -> int:
        """Every allocated block counts in full."""
        return self.allocator.num_used * self.block_size


class PrefixCachingKVCache(PagedKVCache):
    """A paged KV cache that shares full blocks between sequences with a common prefix.

    Each full block is identified by :func:`block_hash`, chained over the blocks
    before it, so equal hashes mean equal prefixes and therefore
    equal KV. A new
    sequence reuses the longest run of cached blocks at its start and only
    prefills the rest. Full blocks are never written again, so sharing them needs
    no copy-on-write; a sequence's partial last block is always private.

    Freed blocks stay cached as evictable free space, so a later request with the
    same system prompt, or a preempted request coming back, can find them again.
    """

    prefix_caching = True

    def __init__(self, pool: KVPool, block_size: int = 16, watermark: float = 0.01) -> None:
        super().__init__(pool, block_size, watermark)
        self.allocator: CachingBlockAllocator = CachingBlockAllocator(self.allocator.num_blocks)
        self._block_hashes: dict[
            int, list[BlockHash]
        ] = {}  # hashes of each sequence's cached blocks
        self._stats = PrefixCacheStats()

    def _match(self, token_ids: Sequence[int]) -> tuple[list[int], list[int]]:
        """Blocks and hashes of the longest cached prefix of ``token_ids``.

        The last token is never matched: it must be prefilled to produce the
        logits for the next token.
        """
        blocks: list[int] = []
        hashes: list[BlockHash] = []
        parent = None
        bs = self.block_size
        for i in range((len(token_ids) - 1) // bs):
            parent = block_hash(parent, token_ids[i * bs : (i + 1) * bs])
            block = self.allocator.lookup(parent)
            if block is None:
                break
            blocks.append(block)
            hashes.append(parent)
        return blocks, hashes

    def match_prefix(self, token_ids: Sequence[int]) -> int:
        """Leading tokens of ``token_ids`` whose KV is cached (a multiple of the block size)."""
        return len(self._match(token_ids)[0]) * self.block_size

    def can_allocate(self, num_tokens: int, token_ids: Sequence[int] | None = None) -> bool:
        """Like :meth:`PagedKVCache.can_allocate`, but cache hits in use cost no memory."""
        needed = self.blocks_needed(num_tokens)
        if token_ids is not None:
            hits, _ = self._match(token_ids)
            # An evictable hit leaves the free pool when acquired, so it is not free.
            needed -= sum(self.allocator.ref_counts[b] > 0 for b in hits)
        return needed <= self.allocator.num_free - self.watermark_blocks

    def _allocate(self, seq_id: int, num_tokens: int, token_ids: Sequence[int] | None) -> bool:
        hits, hashes = self._match(token_ids) if token_ids is not None else ([], [])
        for block in hits:  # before allocating, so the hits cannot be evicted
            self.allocator.acquire(block)
        num_fresh = self.blocks_needed(num_tokens) - len(hits)
        self.block_tables[seq_id] = hits + [self.allocator.allocate() for _ in range(num_fresh)]
        self._block_hashes[seq_id] = hashes
        self._stats.query_tokens += num_tokens
        self._stats.hit_tokens += len(hits) * self.block_size
        return True

    def cache_blocks(self, seq_id: int, token_ids: Sequence[int]) -> None:
        """Register the full blocks of ``token_ids`` (whose KV is computed) in the cache."""
        hashes = self._block_hashes[seq_id]
        num_full = len(token_ids) // self.block_size
        if num_full <= len(hashes):
            return
        table = self.block_tables[seq_id]
        parent = hashes[-1] if hashes else None
        bs = self.block_size
        for i in range(len(hashes), num_full):
            parent = block_hash(parent, token_ids[i * bs : (i + 1) * bs])
            self.allocator.register(table[i], parent)
            hashes.append(parent)

    def _free(self, seq_id: int) -> None:
        # Tail first, so it is evicted before the more widely shared head.
        for block in reversed(self.block_tables.pop(seq_id)):
            self.allocator.free(block)
        del self._block_hashes[seq_id]

    def prefix_cache_stats(self) -> PrefixCacheStats:
        """Hit-rate and eviction counters since the cache was created."""
        self._stats.evicted_blocks = self.allocator.num_evictions
        return self._stats

    def usage(self) -> KVUsage:
        """Like the base usage, but a block shared by several sequences counts once."""
        shared = self.allocator.num_shared_refs * self.block_size
        return KVUsage(
            used_tokens=sum(self._lengths.values()) - shared,
            reserved_tokens=self.reserved_tokens(),
            capacity_tokens=self.pool.num_slots,
            cached_tokens=self.allocator.num_evictable * self.block_size,
        )
