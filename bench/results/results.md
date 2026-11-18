| Strategy | Throughput (tok/s) | vs. naive | p50 latency | p95 latency | p50 TTFT | p95 TTFT | Mean batch | KV utilization |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive (taxi) | 482 | 1.0x | 16.81 s | 29.63 s | 16,628 ms | 29,133 ms | 1.0 | 35% |
| Static batching (bus) | 1,394 | 2.9x | 4.28 s | 7.23 s | 4,082 ms | 6,586 ms | 8.0 | 39% |
| Continuous batching | 1,865 | 3.9x | 2.76 s | 4.42 s | 2,130 ms | 3,638 ms | 14.9 | 35% |
| Continuous + paged KV | 2,271 | 4.7x | 1.59 s | 3.33 s | 676 ms | 1,611 ms | 26.8 | 96% |
| + fused C++ attention | 2,883 | 6.0x | 0.74 s | 2.30 s | 116 ms | 468 ms | 26.4 | 96% |
| + speculative decoding | 2,119 | 4.4x | 2.01 s | 4.09 s | 868 ms | 2,302 ms | 28.3 | 96% |

Hardware: Intel(R) Xeon(R) Processor @ 2.10GHz, 4 logical cores, 4 PyTorch threads, PyTorch 2.9.1+cpu, Python 3.13.18. Commit `c05ea46`, 2026-11-18T05:17:48+00:00.

Throughput of each repeat (tok/s): Naive (taxi) 470/482/497, Static batching (bus) 1,404/1,358/1,394, Continuous batching 1,865/1,840/1,876, Continuous + paged KV 2,312/2,168/2,271, + fused C++ attention 2,883/2,834/2,992, + speculative decoding 2,051/2,119/2,258.
