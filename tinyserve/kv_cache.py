"""KV cache memory: one physical pool, and managers that decide who owns which slots.

During decoding, every new token attends to the keys and values of all previous
tokens in its sequence. Recomputing them each step would make generation
quadratic, so we cache them. The cache is by far the largest piece of serving
memory, and *how it is laid out* decides how many requests fit at once.

This module splits the problem in two:

* :class:`KVPool` is dumb storage: per layer, one flat tensor of ``num_slots``
  token slots, each holding a key and a value of shape ``(n_heads, head_dim)``.
* A :class:`KVCacheManager` owns the bookkeeping: which slots belong to which
  sequence. It turns "token ``j`` of sequence ``s``" into a physical slot address.

The only thing that differs between memory layouts is that address function:

* :class:`ContiguousKVCache` gives each sequence one slab of ``max_seq_len``
  slots up front, so ``address = row * max_seq_len + j``.
* :class:`PagedKVCache` hands out small fixed-size blocks on demand, like the
  pages of virtual memory (and vLLM's PagedAttention), so
  ``address = block_table[j // B] * B + j % B``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor


class KVPool:
    """Physical KV storage: ``k[layer]`` and ``v[layer]`` of shape ``(num_slots, H, D)``."""

    def __init__(
        self,
        num_slots: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_slots = num_slots
        shape = (n_layers, num_slots, n_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype)
        self.v = torch.zeros(shape, dtype=dtype)

    @property
    def nbytes(self) -> int:
        """Total bytes held by the pool (keys + values, all layers)."""
        return self.k.nbytes + self.v.nbytes


@dataclass
class KVBatch:
    """Per-forward-pass instructions telling attention where to write and read KV.

    Attributes:
        pool: the physical storage.
        write_mask: ``(B, S)`` bool, True for real (non-padding) new tokens.
        write_slots: ``(N,)`` slot address for each True entry of ``write_mask``,
            in row-major order.
        read_slots: ``(B, L)`` slot address of token ``j`` of each sequence, for
            ``j < L``. Entries beyond a sequence's length point at harmless
            garbage; the causal mask in attention hides them.
    """

    pool: KVPool
    write_mask: Tensor
    write_slots: Tensor
    read_slots: Tensor

    def write_and_read(self, layer: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Store new ``(B, S, H, D)`` keys/values, then gather the ``(B, L, H, D)`` history."""
        k_cache, v_cache = self.pool.k[layer], self.pool.v[layer]
        k_cache[self.write_slots] = k[self.write_mask]
        v_cache[self.write_slots] = v[self.write_mask]
        return k_cache[self.read_slots], v_cache[self.read_slots]


@dataclass(frozen=True)
class KVUsage:
    """A snapshot of KV memory usage, in tokens.

    Attributes:
        used_tokens: slots holding a real token's KV right now.
        reserved_tokens: slots handed out to sequences (used + reserved-but-empty).
        capacity_tokens: total slots in the pool.
    """

    used_tokens: int
    reserved_tokens: int
    capacity_tokens: int

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


class KVCacheManager(ABC):
    """Bookkeeping for which pool slots belong to which sequence.

    The engine calls :meth:`allocate` when a sequence is admitted, :meth:`grow`
    before each step that adds tokens, and :meth:`free` when it finishes or is
    preempted. :meth:`build_batch` turns a batch of sequences and positions into
    the :class:`KVBatch` the model consumes.
    """

    layout = "base"

    def __init__(self, pool: KVPool) -> None:
        self.pool = pool
        self._lengths: dict[int, int] = {}

    # --- allocation -------------------------------------------------------------

    @abstractmethod
    def can_allocate(self, num_tokens: int) -> bool:
        """Whether a *new* sequence needing ``num_tokens`` slots would fit right now."""

    def allocate(self, seq_id: int, num_tokens: int) -> bool:
        """Reserve memory for a new sequence of ``num_tokens``. Returns False if it won't fit."""
        if seq_id in self._lengths:
            raise ValueError(f"sequence {seq_id} is already allocated")
        if not self.can_allocate(num_tokens) or not self._allocate(seq_id, num_tokens):
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

    @abstractmethod
    def _allocate(self, seq_id: int, num_tokens: int) -> bool: ...

    @abstractmethod
    def _grow(self, seq_id: int, num_tokens: int) -> bool: ...

    @abstractmethod
    def _free(self, seq_id: int) -> None: ...

    # --- addressing -------------------------------------------------------------

    @abstractmethod
    def addresses(self, seq_ids: list[int], positions: Tensor) -> Tensor:
        """Map ``positions[b, j]`` of sequence ``seq_ids[b]`` to physical slot indices."""

    def build_batch(self, seq_ids: list[int], positions: Tensor, valid: Tensor) -> KVBatch:
        """Build the :class:`KVBatch` for one forward pass.

        Args:
            seq_ids: the sequences in the batch, one per row.
            positions: ``(B, S)`` absolute positions of the new tokens.
            valid: ``(B, S)`` bool, False for padding.
        """
        write_slots = self.addresses(seq_ids, positions)[valid]
        context_len = int(positions[valid].max()) + 1
        all_positions = torch.arange(context_len).expand(len(seq_ids), -1)
        read_slots = self.addresses(seq_ids, all_positions)
        return KVBatch(self.pool, valid, write_slots, read_slots)

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
        super().__init__(pool)
        self.max_seq_len = max_seq_len
        self.num_rows = pool.num_slots // max_seq_len
        if self.num_rows == 0:
            raise ValueError("pool is smaller than a single max_seq_len slab")
        self._free_rows = list(range(self.num_rows - 1, -1, -1))  # pop() yields row 0 first
        self._row_of: dict[int, int] = {}

    def can_allocate(self, num_tokens: int) -> bool:
        """A new sequence fits if any slab is free and it is not longer than a slab."""
        return bool(self._free_rows) and num_tokens <= self.max_seq_len

    def _allocate(self, seq_id: int, num_tokens: int) -> bool:
        self._row_of[seq_id] = self._free_rows.pop()
        return True

    def _grow(self, seq_id: int, num_tokens: int) -> bool:
        return num_tokens <= self.max_seq_len

    def _free(self, seq_id: int) -> None:
        self._free_rows.append(self._row_of.pop(seq_id))

    def row_of(self, seq_id: int) -> int:
        """The slab index owned by ``seq_id``."""
        return self._row_of[seq_id]

    def addresses(self, seq_ids: list[int], positions: Tensor) -> Tensor:
        """``row * max_seq_len + position``."""
        rows = torch.tensor([self._row_of[s] for s in seq_ids], dtype=torch.long)
        return rows[:, None] * self.max_seq_len + positions

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
        return self.num_blocks - len(self._free)

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
        super().__init__(pool)
        self.block_size = block_size
        self.allocator = BlockAllocator(pool.num_slots // block_size)
        self.watermark_blocks = int(watermark * self.allocator.num_blocks)
        self.block_tables: dict[int, list[int]] = {}

    def blocks_needed(self, num_tokens: int) -> int:
        """Number of blocks that hold ``num_tokens`` tokens."""
        return -(-num_tokens // self.block_size)  # ceil division

    def can_allocate(self, num_tokens: int) -> bool:
        """A new sequence fits if its blocks are free without dipping into the watermark."""
        needed = self.blocks_needed(num_tokens)
        return needed <= self.allocator.num_free - self.watermark_blocks

    def _allocate(self, seq_id: int, num_tokens: int) -> bool:
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

    def addresses(self, seq_ids: list[int], positions: Tensor) -> Tensor:
        """``block_table[pos // block_size] * block_size + pos % block_size``."""
        tables = [self.block_tables[s] for s in seq_ids]
        width = max(len(t) for t in tables)
        padded = torch.zeros(len(tables), width, dtype=torch.long)
        for i, t in enumerate(tables):
            padded[i, : len(t)] = torch.tensor(t)
        logical_block = (positions // self.block_size).clamp(max=width - 1)
        return padded.gather(1, logical_block) * self.block_size + positions % self.block_size

    def reserved_tokens(self) -> int:
        """Every allocated block counts in full."""
        return self.allocator.num_used * self.block_size
