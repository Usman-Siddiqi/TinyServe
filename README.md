<div align="center">

# 🚐 TinyServe

**A tiny LLM inference engine, built from scratch to show how production model serving actually works.**

Continuous batching · paged KV cache · request scheduling · token streaming. In under 2,000 lines of heavily commented Python.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/Usman-Siddiqi/TinyServe/actions/workflows/ci.yml/badge.svg)](https://github.com/Usman-Siddiqi/TinyServe/actions/workflows/ci.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

</div>

> [!NOTE]
> The model inside TinyServe has **random weights** (fixed seed), so everything it writes is gibberish. That is on purpose. This project is about the *serving system* around a model: how requests are queued, batched, scheduled, given memory and streamed back. A real checkpoint would make the output readable but would not change a single line of the engine.

<p align="center">
  <img src="docs/assets/batching.gif" alt="Static batching leaves seats empty until the whole batch finishes; continuous batching refills every seat at every step" width="900">
</p>

## Why this exists

Serving an LLM to many users at once is mostly a scheduling and memory problem, not a math problem. The math (one transformer forward pass) is the same for everyone. What differs between a toy and a production server like vLLM or TGI is how it decides **who gets to run on the next step** and **where their memory lives**.

Think of the GPU (here, the CPU) as a vehicle carrying passengers (requests) who each travel a different distance (number of tokens to generate):

| | Strategy | Analogy | What goes wrong |
|---|---|---|---|
| 🚕 | **Naive** | A **taxi**: one passenger, door to door. | Everyone else waits at the curb. The hardware does one small job at a time and sits mostly idle. |
| 🚌 | **Static batching** | A **bus that waits for everyone**: it fills up, drives, and nobody new can board until the last passenger gets off. | Short trips finish early and their seats ride empty until the longest trip is done. New arrivals wait for the whole bus to come back. |
| 🚐 | **Continuous batching** | A **shuttle that stops at every step**: people get off the moment they arrive, and whoever is waiting takes the free seat right away. | Nothing, as long as there is room. Which is where the KV cache comes in. |

Every token a request has seen must stay in memory (the **KV cache**) until it finishes. If you reserve that memory the simple way, one big slab sized for the longest possible request, most of it sits empty and only a handful of passengers fit in the shuttle. A **paged KV cache** hands out memory in small fixed-size blocks as requests actually grow, like an operating system's virtual memory pages. Same memory, roughly twice the passengers.

TinyServe implements all of that from scratch, measures it, and draws it.

## Results

The benchmark fires the **same 200 requests** (Poisson arrivals, random prompt lengths of 16 to 192 characters and long-tailed output lengths of 10 to 300 tokens, fixed seed) at four engine configurations that share the **same model and the same KV memory budget** (8,192 token slots).

<p align="center"><img src="docs/assets/throughput.png" alt="Throughput by strategy" width="760"></p>

| Strategy | Throughput (tok/s) | vs. naive | p50 latency | p95 latency | p50 TTFT | p95 TTFT | Mean batch | KV utilization |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive (taxi) | 428 | 1.0x | 18.72 s | 33.79 s | 18,450 ms | 33,214 ms | 1.0 | 35% |
| Static batching (bus) | 1,152 | 2.7x | 5.66 s | 9.61 s | 5,376 ms | 8,681 ms | 7.8 | 40% |
| Continuous batching | 1,646 | 3.8x | 3.15 s | 5.37 s | 2,279 ms | 4,434 ms | 14.7 | 36% |
| Continuous + paged KV | 2,115 | 4.9x | 2.10 s | 3.95 s | 1,059 ms | 2,239 ms | 27.7 | 96% |

Hardware: Intel(R) Xeon(R) Processor @ 2.30GHz, 4 logical cores, 4 PyTorch threads, PyTorch 2.9.1+cpu, Python 3.13.18. Commit `049f63c`, 2026-11-17T23:54:41+00:00.

Throughput of each repeat (tok/s): Naive (taxi) 406/477/428, Static batching (bus) 1,111/1,347/1,152, Continuous batching 1,866/1,636/1,646, Continuous + paged KV 2,283/1,862/2,115.

**What the numbers say**

- **Continuous vs. static batching: 1.4x the throughput** (1,646 vs. 1,152 tok/s) with the *same* 16-request memory limit, and p50 latency drops from 5.7 s to 3.2 s. The static batcher averages only 7.8 busy seats out of 16, because finished requests leave empty seats that cannot be refilled until the whole batch drains.
- **Paging the KV cache: another 1.3x** (2,115 tok/s) **and half the time-to-first-token** (p50 2.3 s to 1.1 s). Nothing about the scheduler changed; the same 8,192-slot pool simply fits 27.7 running requests on average instead of 14.7, because 96% of reserved memory holds real tokens instead of 36%.
- **End to end: 4.9x the throughput of serving one request at a time**, and the median request finishes in 2.1 s instead of 18.7 s.
- **The trade-off is real**: a bigger batch makes each decode step slower (median time per output token goes from 5.7 ms with static batching to 13.7 ms with paging). Requests still finish much sooner because they spend far less time waiting in the queue. Production servers tune `max_batch_size` and the token budget exactly along this curve.
- **Run-to-run noise** on the shared cloud VM is around ±10%; every repeat is listed under the table and stored in the JSON.

<p align="center"><img src="docs/assets/latency_cdf.png" alt="Latency and time-to-first-token CDFs" width="900"></p>

<p align="center"><img src="docs/assets/batch_size.png" alt="Running batch size over time" width="860"></p>

The static batcher's sawtooth is the bus problem in one picture: the batch fills, drains as passengers get off, and sits half empty waiting for the longest request before anyone new can board. Continuous batching stays pinned at its limit, and with paging the limit itself doubles.

### Paged vs. contiguous KV memory

<p align="center">
  <img src="docs/assets/paged_kv.gif" alt="A contiguous KV cache reserves a 512-token slab per request and wastes most of it; a paged cache allocates 16-token blocks on demand" width="900">
</p>

<p align="center"><img src="docs/assets/kv_memory.png" alt="Reserved vs. used KV memory over time" width="900"></p>

With contiguous slabs, a request that ends up 120 tokens long still pins 512 slots for its whole life, so the 8,192-slot pool holds only 16 requests and most of what is reserved is empty. With 16-token pages a request only ever wastes the unused tail of its last block.

### Scheduling for goodput

Throughput is not what users feel. Production servers are judged on **goodput**: requests per second that meet a latency SLO, here a time to first token (TTFT) and a time per output token (TPOT) target. Once the shuttle is full, *who boards next* decides how many passengers arrive on time. The continuous scheduler takes a pluggable queue policy (`tinyserve/policy.py`):

| Policy | Admits first | Preempts first |
|---|---|---|
| `fcfs` (default) | oldest | newest |
| `sjf` | smallest prompt + `max_new_tokens`, with aging so big jobs cannot starve | largest |
| `edf` | earliest TTFT deadline; requests already past it go last | latest deadline |
| `slo` | *shortest feasible job*: smallest job among those that can still make their deadline; late ones last, oldest first | largest |

<p align="center"><img src="docs/assets/goodput.png" alt="Goodput, SLO attainment and latency cost of each queue policy from half to three times capacity" width="900"></p>

The chart is from the scheduler simulator (below): 1,000 requests per point, median of 5 seeds, SLO of TTFT <= 3 s and TPOT <= 150 ms. The sweep ran with 1 PyTorch thread on a shared, busy VM, where the engine served 445 tok/s (about 5x slower than the main benchmark), so the SLO is scaled up to match. Load is a multiple of the capacity measured by serving a burst of requests.

- **Below capacity the policy does not matter.** The queue is empty, so every policy makes the same choices.
- **Past capacity FCFS collapses.** Every request queues behind older ones that are already late, so at 1.5x load only 13% meet the SLO.
- **Demoting late requests is most of the fix.** `edf` keeps 66% on time at 1.5x. Adding shortest-first among the on-time ones (`slo`) keeps 88%: **goodput 6.9x FCFS at 1.5x load and 10x at 2x**.
- **SJF alone barely helps** (+10% at 1.5x). Short jobs finish sooner, but everyone is still late.
- **The price is the tail.** Demoted requests wait until the burst passes: p99 latency goes from 55 s to 126 s at 1.5x, and with `slo` the longest 10% of jobs wait about twice as long (median 37 s to 68 s). `edf` is the fairer option: at 1.5x its long jobs even finish sooner than under FCFS (22 s). A production server would pair either one with load shedding, so a request that is already late gets a fast "busy" answer instead of a long wait.
- **The size hint is an oracle here.** Benchmarks ignore EOS, so `max_new_tokens` is the exact output length. Real servers only know an upper bound or a predicted length.

**Real engine, heavy overload** (4x load, 200 requests, 3 interleaved repeats): `slo` beat FCFS in every repeat, median goodput **2.0x** (3.51 vs. 1.72 req/s) and 58% vs. 26% of requests on time. The median request finished in 6.9 s instead of 12.9 s, while p99 latency rose from 21.3 s to 25.7 s and the longest 10% of jobs from 18.0 s to 22.3 s. The raw runs are in [`bench/results/sweep.json`](bench/results/sweep.json).

#### A simulator for scheduling experiments

On a shared VM, the engine's speed drifted by up to about 30% between runs, and near capacity that alone swung FCFS from 78% to 100% of requests on time in two repeats of the same workload. Scheduling decisions do not depend on token values, so `tinyserve/simulator.py` replays a workload through the *real* scheduler and KV cache manager on a virtual clock, charging each step a duration from a linear cost model (fixed overhead, prefill tokens, decode context tokens) fitted to a real engine trace. A 1,000-request run takes under a second and is deterministic.

To check it, every real sweep run is replayed with a cost model fitted to its own trace. Across 22 runs the simulator's goodput is within 1.4% of the measured one (median; worst 31%), and p99 latency within 2.0% (worst 40%). The misses are single runs; one linear model per run cannot follow a machine whose speed drifts within the run.

<p align="center"><img src="docs/assets/sim_check.png" alt="Simulated vs. measured SLO attainment, p99 TTFT and p99 latency for 22 real runs" width="900"></p>

## Architecture

```mermaid
flowchart LR
    C["Clients<br/>(curl, load generator)"] -- "POST /generate<br/>SSE token stream" --> API["FastAPI server<br/>server.py"]
    API -- "GET /stats" --> C
    API <--> AE["AsyncEngine<br/>inbox + per-request queues<br/>async_engine.py"]
    AE --> E["Engine loop<br/>engine.py"]
    E --> S["Scheduler<br/>naive / static / continuous<br/>scheduler.py"]
    S --> KV["KV cache manager<br/>contiguous or paged blocks<br/>kv_cache.py"]
    E --> R["Model runner<br/>prefill + decode batches<br/>runner.py"]
    R --> M["TinyGPT<br/>model.py"]
    M -- "write / gather K,V<br/>by slot address" --> KV
```

One engine step does three things: the **scheduler** picks who runs (admitting, finishing and preempting requests while asking the **KV cache manager** for memory), the **model runner** executes one batched prefill and/or decode pass, and each request receives its new token, which the **AsyncEngine** pushes to the right HTTP stream.

## Quickstart

```bash
git clone https://github.com/Usman-Siddiqi/TinyServe.git && cd TinyServe
python -m venv .venv && source .venv/bin/activate

# CPU-only PyTorch (small download), then TinyServe itself
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,bench]"

pytest                      # 60 tests, a few seconds
```

**Run the server** (continuous batching with a paged KV cache by default):

```bash
tinyserve-server --port 8000
# or: tinyserve-server --strategy static --max-batch-size 16
```

**Stream a completion** over Server-Sent Events:

```bash
curl -N -X POST localhost:8000/generate \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "Hello, TinyServe!", "max_tokens": 8, "temperature": 1.0, "seed": 0}'
```

```text
data: {"index": 0, "token": "f", "token_id": 74}
data: {"index": 1, "token": "_", "token_id": 67}
data: {"index": 2, "token": "M", "token_id": 49}
...
data: {"finished": true, "text": "f_Me<11", "finish_reason": "length", "num_prompt_tokens": 18, "num_generated_tokens": 8, "ttft_ms": 15.96, "latency_ms": 104.22, "num_preemptions": 0}
```

Gibberish, as promised: the weights are random.

**Check what the engine is doing** (this snapshot was taken while the load generator below was running):

```bash
curl -s localhost:8000/stats
```

```json
{
    "strategy": "continuous",
    "kv_layout": "paged",
    "queue_depth": 0,
    "batch_size": 15,
    "max_batch_size": 32,
    "kv_used_tokens": 3085,
    "kv_reserved_tokens": 3200,
    "kv_capacity_tokens": 8192,
    "kv_utilization": 0.9641,
    "kv_occupancy": 0.3906,
    "num_steps": 378,
    "num_generated_tokens": 3549,
    "num_finished_requests": 32,
    "num_preemptions": 0,
    "tokens_per_sec": 708.2,
    "uptime_sec": 8.0
}
```

**Load test it** with the async load generator (Poisson arrivals, streams every response):

```bash
tinyserve-loadgen --url http://127.0.0.1:8000 --num-requests 100 --rate 10
```

**Reproduce the benchmark and charts** (about 4 minutes on a 4-core machine):

```bash
python bench/run.py          # writes bench/results/results.json (raw, per request)
python bench/plot.py         # renders docs/assets/*.png and bench/results/results.md
python docs/make_visuals.py  # renders the two GIFs from real engine traces
python bench/sweep.py --policies fcfs slo     # goodput load sweep (real engine)
python bench/sweep.py --mode sim --calibration bench/results/sweep.json \
    --out bench/results/sweep_sim.json --policies fcfs sjf edf slo --num-requests 1000 --seeds 0 1 2 3 4
```

## Project structure

```text
tinyserve/
  model.py         TinyGPT: decoder-only transformer whose attention reads an external KV cache
  tokenizer.py     printable-ASCII character tokenizer
  kv_cache.py      KV pool, contiguous and paged cache managers, block allocator
  request.py       requests, sampling params, per-request timing metrics
  sampler.py       greedy / temperature sampling with per-request RNGs
  runner.py        builds padded prefill and decode batches, runs the model
  scheduler.py     naive, static and continuous batching schedulers (+ preemption)
  policy.py        queue policies: FCFS, SJF, EDF, shortest feasible job first
  simulator.py     replays a workload through a scheduler on a virtual clock
  engine.py        the step loop, tracing, and build_engine() from an EngineConfig
  async_engine.py  background engine thread <-> asyncio bridge
  server.py        FastAPI app: POST /generate (SSE), GET /stats, GET /health
  workload.py      reproducible Poisson workloads
  loadgen.py       async HTTP load generator
bench/
  run.py           benchmark all strategies, save raw JSON
  plot.py          charts + results tables from the JSON files
  sweep.py         goodput vs. load sweep of the queue policies
  results/         the raw results behind every number in this README
docs/
  make_visuals.py  animated GIFs replayed from real engine traces
  assets/          generated charts and GIFs
tests/             pytest suite, one file per component
```

## How it works

### The model (`model.py`)

A GPT-style decoder: 4 layers, 8 query heads sharing 2 key/value heads (grouped-query attention, as in Llama 2/3 and Mistral), `d_model` 512, an MLP of width 2048 and learned positional embeddings. About 11M parameters, a 101-token character vocabulary and a 512-token context. The weights come from a fixed seed under a private RNG, so the model is bit-for-bit identical everywhere.

The important design choice is that **attention does not own its history**. Each forward pass receives a `KVBatch` that says, for every new token, which slot of a shared memory pool to write its key/value to, and for every sequence, which slots hold its past. The causal mask is computed from absolute positions (`key_pos <= query_pos`), which also hides padding and stale cache entries for free. Tests check that incremental decoding with the cache matches a full recompute, and that ragged batches give the same logits as running each sequence alone.

### Prefill vs. decode (`runner.py`)

A request's life has two phases. **Prefill** pushes its whole prompt through the model in one pass, fills the cache and samples the first token: it is compute-heavy and sets time-to-first-token. **Decode** feeds one token per request per step: each step is cheap, so it is the phase that benefits most from batching many requests together. Every scheduler uses the same runner, so the comparison only measures scheduling and memory policy.

### Schedulers (`scheduler.py`)

All three share a FIFO waiting queue and a running list, and differ only in when they let requests board:

- **`NaiveScheduler`** runs one request start to finish.
- **`StaticScheduler`** boards up to `max_batch_size` requests when the batch is empty and runs them until all are done. It is given a head start: finished requests are dropped from the compute rather than decoded as padding.
- **`ContinuousScheduler`** re-plans at every step. It first reserves KV room for every running request's next token, preempting the newest request if memory runs out. Then it admits waiting requests in FIFO order (or the order of another [queue policy](#scheduling-for-goodput)) while there is a free seat, the step's token budget (`max_tokens_per_step`, which bounds how long a burst of long prompts can stall everyone else) allows it, and their KV memory fits. Newcomers are prefilled in the same step the others decode.

Preemption uses **recompute**: the victim's memory is freed and it goes back to the front of the queue keeping the tokens it generated. On readmission its whole sequence is prefilled again, which costs compute but needs no extra memory (vLLM's default strategy). A test squeezes six requests into memory for two and checks that outputs are unchanged.

### KV cache (`kv_cache.py`)

The pool is plain storage, per layer a `(num_slots, n_kv_heads, head_dim)` tensor for keys and one for values. Managers decide which slots belong to which sequence, and the two layouts differ only in their address function:

| Layout | Reserves | Address of token `j` of a sequence |
|---|---|---|
| `ContiguousKVCache` | a `max_seq_len` slab per request, up front | `row * max_seq_len + j` |
| `PagedKVCache` | 16-token blocks, on demand | `block_table[j // 16] * 16 + j % 16` |

`BlockAllocator` is a free list of block ids. Because every block has the same size there is no external fragmentation and nothing to compact, and a freed block is immediately reusable by anyone. Each sequence keeps a block table mapping logical blocks to physical ones, which can be scattered anywhere in the pool. `KVUsage` reports used slots (holding a real token), reserved slots (handed out) and capacity, so utilization is `used / reserved`.

Attention gathers each sequence's keys and values through those addresses into a dense tensor. Rows are sorted by length and attended in groups of 8 so a 20-token sequence is not padded out to a 450-token one in the same batch.

### Server (`async_engine.py`, `server.py`)

The engine is synchronous PyTorch, so it runs in one background thread looping `engine.step()`. HTTP handlers on the asyncio loop talk to it through a thread-safe **inbox** (add / abort commands) and one `asyncio.Queue` per request that the engine thread fills via `loop.call_soon_threadsafe`. Only the engine thread touches the engine, so it needs no locks. `POST /generate` streams each token as an SSE `data:` event and ends with a summary event (finish reason, TTFT, latency). If the client disconnects mid-stream, the request is aborted and its KV blocks are freed. `GET /stats` reports queue depth, batch size, KV utilization and occupancy, preemptions and a rolling tokens/sec.

### Benchmark (`bench/`)

`bench/run.py` builds a fresh engine per configuration, warms it up, then injects the workload in real time at the Poisson arrival times, so queueing delay counts toward latency exactly as it would for users. The engine is driven in-process (no HTTP), so the numbers measure the serving system rather than JSON and sockets. Each configuration runs three times, interleaved, and the median-throughput run is reported. All raw per-request and per-step data is in [`bench/results/results.json`](bench/results/results.json).

## Limitations

- **CPU only.** Everything runs on PyTorch's CPU kernels. Absolute numbers are tiny next to a GPU, and the shared cloud VM the benchmark ran on adds a few percent of run-to-run noise (all repeats are listed under the table).
- **Random weights.** The output is noise. Under greedy decoding the model tends to repeat one character; use `temperature` for more varied noise.
- **Simplified attention.** There is no fused paged-attention kernel: each step gathers a sequence's K/V from its blocks into a temporary dense tensor and calls PyTorch's `scaled_dot_product_attention`. That extra copy is the main cost of decoding long sequences in big batches, and it is why nearly doubling the batch with paging buys 1.3x more throughput here rather than 2x. Prefill batches are padded to their longest prompt.
- **A toy model and tokenizer.** 11M parameters, a character vocabulary, learned positions instead of RoPE, float32 throughout.
- **No chunked prefill or prefill/decode disaggregation.** A long prompt is prefilled in one go (bounded by the per-step token budget).

## Roadmap

- [ ] **Prefix caching**: share KV blocks between requests with a common prompt prefix (block hashing plus reference counts in the allocator).
- [ ] **Speculative decoding**: a smaller draft model proposes several tokens and the main model verifies them in one pass.
- [ ] **C++ paged-attention kernel**: read K/V straight from the blocks instead of gathering them into a temporary tensor.
- [ ] **Preemption by swapping**: move a victim's blocks to a CPU-side pool instead of recomputing them.
- [ ] Chunked prefill, so long prompts never stall decoding.
- [ ] A bf16 KV cache to halve memory traffic.

## License

[MIT](LICENSE) © Usman Siddiqi
