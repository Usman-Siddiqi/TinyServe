"""Synthetic workloads with Poisson arrivals, shared by the load generator and benchmarks."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

# Printable characters used to build random prompts (the model does not care what they say).
_ALPHABET = "abcdefghijklmnopqrstuvwxyz      ,.ETAOINSHRDLU"


@dataclass(frozen=True)
class WorkloadItem:
    """One request in a workload.

    Attributes:
        arrival: seconds after the start of the run at which the request is sent.
        prompt: prompt text.
        max_tokens: exact number of tokens to generate (benchmarks ignore EOS).
    """

    arrival: float
    prompt: str
    max_tokens: int

    def to_dict(self) -> dict[str, float | str | int]:
        """JSON-friendly form."""
        return asdict(self)


@dataclass(frozen=True)
class WorkloadSpec:
    """Parameters of a synthetic workload.

    Output lengths follow a clipped log-normal distribution: most answers are
    short, a few are very long. That long tail is exactly what hurts static
    batching, where everyone waits for the longest request in the batch.

    A ``long_prompt_fraction`` of the requests instead get a prompt of
    ``long_prompt_len`` characters (e.g. pasted documents), with the output
    clipped so the whole sequence fits in ``max_total_len``.
    """

    num_requests: int = 200
    rate: float = 20.0  # mean requests per second (Poisson process)
    seed: int = 0
    min_prompt_len: int = 16
    max_prompt_len: int = 192
    median_output_len: int = 64
    output_len_sigma: float = 0.8
    min_output_len: int = 8
    max_output_len: int = 300
    long_prompt_fraction: float = 0.0
    long_prompt_len: tuple[int, int] = (300, 450)
    max_total_len: int = 500  # leaves room for BOS under max_seq_len=512


def make_workload(spec: WorkloadSpec) -> list[WorkloadItem]:
    """Generate a reproducible list of requests sorted by arrival time."""
    rng = random.Random(spec.seed)
    t = 0.0
    items = []
    for _ in range(spec.num_requests):
        t += rng.expovariate(spec.rate)  # exponential gaps = Poisson arrivals
        # Only draw the extra random number when enabled, so default workloads stay identical.
        if spec.long_prompt_fraction and rng.random() < spec.long_prompt_fraction:
            prompt_len = rng.randint(*spec.long_prompt_len)
        else:
            prompt_len = rng.randint(spec.min_prompt_len, spec.max_prompt_len)
        prompt = "".join(rng.choice(_ALPHABET) for _ in range(prompt_len))
        out = rng.lognormvariate(math.log(spec.median_output_len), spec.output_len_sigma)
        longest = min(spec.max_output_len, spec.max_total_len - prompt_len)
        max_tokens = int(min(max(out, spec.min_output_len), longest))
        items.append(WorkloadItem(arrival=t, prompt=prompt, max_tokens=max_tokens))
    return items
