| Strategy | Throughput (tok/s) | vs. naive | p50 latency | p95 latency | p50 TTFT | p95 TTFT | Mean batch | KV utilization |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive (taxi) | 428 | 1.0x | 18.72 s | 33.79 s | 18,450 ms | 33,214 ms | 1.0 | 35% |
| Static batching (bus) | 1,152 | 2.7x | 5.66 s | 9.61 s | 5,376 ms | 8,681 ms | 7.8 | 40% |
| Continuous batching | 1,646 | 3.8x | 3.15 s | 5.37 s | 2,279 ms | 4,434 ms | 14.7 | 36% |
| Continuous + paged KV | 2,115 | 4.9x | 2.10 s | 3.95 s | 1,059 ms | 2,239 ms | 27.7 | 96% |

Hardware: Intel(R) Xeon(R) Processor @ 2.30GHz, 4 logical cores, 4 PyTorch threads, PyTorch 2.9.1+cpu, Python 3.13.18. Commit `049f63c`, 2026-11-17T23:54:41+00:00.

Throughput of each repeat (tok/s): Naive (taxi) 406/477/428, Static batching (bus) 1,111/1,347/1,152, Continuous batching 1,866/1,636/1,646, Continuous + paged KV 2,283/1,862/2,115.
