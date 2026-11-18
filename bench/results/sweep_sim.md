| Load | Policy | Goodput (x capacity) | SLO met | p50 TTFT | p99 TTFT | p50 latency | p99 latency | Longest 10%: p50 latency |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5x | FCFS | 0.49 | 100% | 0.08 s | 0.19 s | 1.05 s | 5.95 s | 4.10 s |
| 0.5x | Shortest job first | 0.49 | 100% | 0.08 s | 0.19 s | 1.05 s | 5.95 s | 4.10 s |
| 0.5x | Earliest deadline, late ones last | 0.49 | 100% | 0.08 s | 0.19 s | 1.05 s | 5.95 s | 4.10 s |
| 0.5x | Shortest feasible job first | 0.49 | 100% | 0.08 s | 0.19 s | 1.05 s | 5.95 s | 4.10 s |
| 0.8x | FCFS | 0.78 | 100% | 0.10 s | 0.28 s | 1.63 s | 11.12 s | 6.23 s |
| 0.8x | Shortest job first | 0.78 | 100% | 0.10 s | 0.27 s | 1.63 s | 11.12 s | 6.23 s |
| 0.8x | Earliest deadline, late ones last | 0.78 | 100% | 0.10 s | 0.28 s | 1.63 s | 11.12 s | 6.23 s |
| 0.8x | Shortest feasible job first | 0.78 | 100% | 0.10 s | 0.27 s | 1.63 s | 11.12 s | 6.23 s |
| 1x | FCFS | 0.92 | 98% | 0.13 s | 3.19 s | 3.15 s | 18.48 s | 10.35 s |
| 1x | Shortest job first | 0.92 | 99% | 0.13 s | 2.80 s | 2.97 s | 18.36 s | 10.39 s |
| 1x | Earliest deadline, late ones last | 0.95 | 100% | 0.13 s | 2.75 s | 3.13 s | 18.59 s | 10.33 s |
| 1x | Shortest feasible job first | 0.96 | 99% | 0.13 s | 2.20 s | 2.91 s | 18.46 s | 10.39 s |
| 1.25x | FCFS | 0.44 | 41% | 3.96 s | 17.65 s | 13.01 s | 33.88 s | 22.58 s |
| 1.25x | Shortest job first | 0.48 | 45% | 3.74 s | 17.82 s | 12.30 s | 34.66 s | 24.03 s |
| 1.25x | Earliest deadline, late ones last | 0.92 | 86% | 1.80 s | 81.67 s | 6.33 s | 89.92 s | 18.25 s |
| 1.25x | Shortest feasible job first | 1.00 | 93% | 0.94 s | 71.78 s | 5.24 s | 83.86 s | 20.83 s |
| 1.5x | FCFS | 0.14 | 13% | 16.11 s | 44.07 s | 25.22 s | 55.23 s | 36.69 s |
| 1.5x | Shortest job first | 0.15 | 14% | 15.49 s | 43.99 s | 24.40 s | 55.90 s | 37.84 s |
| 1.5x | Earliest deadline, late ones last | 0.70 | 66% | 2.82 s | 118.27 s | 8.01 s | 126.18 s | 22.06 s |
| 1.5x | Shortest feasible job first | 0.94 | 88% | 1.48 s | 115.74 s | 5.63 s | 125.89 s | 68.35 s |
| 2x | FCFS | 0.08 | 7% | 33.84 s | 78.99 s | 40.79 s | 86.41 s | 52.90 s |
| 2x | Shortest job first | 0.10 | 9% | 32.38 s | 78.33 s | 40.46 s | 86.61 s | 55.35 s |
| 2x | Earliest deadline, late ones last | 0.38 | 35% | 3.09 s | 99.35 s | 12.96 s | 109.80 s | 23.83 s |
| 2x | Shortest feasible job first | 0.82 | 77% | 2.18 s | 98.02 s | 6.34 s | 115.34 s | 102.42 s |
| 3x | FCFS | 0.06 | 6% | 51.70 s | 114.00 s | 57.93 s | 119.54 s | 73.20 s |
| 3x | Shortest job first | 0.07 | 6% | 49.95 s | 113.31 s | 56.68 s | 120.21 s | 76.35 s |
| 3x | Earliest deadline, late ones last | 0.19 | 18% | 76.33 s | 115.27 s | 80.30 s | 123.77 s | 88.14 s |
| 3x | Shortest feasible job first | 0.65 | 60% | 2.73 s | 113.09 s | 7.02 s | 125.62 s | 105.99 s |
