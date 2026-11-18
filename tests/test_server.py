"""Phase 6: the HTTP API, SSE streaming and stats."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace

import pytest
from conftest import SMALL_MODEL, reference_generate
from fastapi.testclient import TestClient

from tinyserve.engine import EngineConfig
from tinyserve.model import TinyGPT
from tinyserve.request import SamplingParams
from tinyserve.server import create_app
from tinyserve.tokenizer import CharTokenizer

CONFIG = EngineConfig(
    strategy="continuous", model=SMALL_MODEL, kv_capacity_tokens=1024, kv_layout="paged"
)


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(create_app(CONFIG)) as c:
        yield c


def read_events(resp) -> list[dict]:
    return [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]


def test_generate_streams_tokens_as_sse(client: TestClient, model: TinyGPT) -> None:
    body = {"prompt": "Hello there", "max_tokens": 8, "ignore_eos": True}
    with client.stream("POST", "/generate", json=body) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = read_events(resp)
    tokens, summary = events[:-1], events[-1]
    assert [e["index"] for e in tokens] == list(range(8))
    assert summary["finished"] is True
    assert summary["num_generated_tokens"] == 8
    assert summary["finish_reason"] == "length"
    assert summary["text"] == "".join(e["token"] for e in tokens)

    # Same tokens as a direct, cache-free greedy run of the model.
    prompt_ids = CharTokenizer().encode("Hello there")
    expected = reference_generate(model, prompt_ids, SamplingParams(max_new_tokens=8))
    assert [e["token_id"] for e in tokens] == expected


def test_generate_without_streaming(client: TestClient) -> None:
    body = {"prompt": "abc", "max_tokens": 5, "ignore_eos": True, "stream": False}
    resp = client.post("/generate", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["num_generated_tokens"] == 5
    assert data["ttft_ms"] >= 0


def test_too_long_request_is_rejected(client: TestClient) -> None:
    body = {"prompt": "x" * 100, "max_tokens": SMALL_MODEL.max_seq_len}
    assert client.post("/generate", json=body).status_code == 400


def test_bad_body_is_rejected(client: TestClient) -> None:
    assert client.post("/generate", json={"prompt": "", "max_tokens": 4}).status_code == 422
    assert client.post("/generate", json={"prompt": "a", "max_tokens": 0}).status_code == 422


def test_stats_reports_engine_state(client: TestClient) -> None:
    client.post("/generate", json={"prompt": "a", "max_tokens": 3, "stream": False})
    stats = client.get("/stats").json()
    for key in (
        "queue_depth",
        "batch_size",
        "kv_utilization",
        "kv_used_tokens",
        "kv_capacity_tokens",
        "tokens_per_sec",
        "num_generated_tokens",
    ):
        assert key in stats
    assert stats["strategy"] == "continuous"
    assert stats["kv_layout"] == "paged"
    assert stats["kv_capacity_tokens"] == 1024
    assert stats["num_generated_tokens"] >= 3
    assert stats["batch_size"] == 0  # idle again, all memory returned
    assert stats["kv_used_tokens"] == 0


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_stats_report_prefix_cache_hits() -> None:
    config = replace(CONFIG, enable_prefix_caching=True)
    body = {"prompt": "You are a helpful assistant. " * 2, "max_tokens": 2, "stream": False}
    with TestClient(create_app(config)) as c:
        first = c.post("/generate", json=body).json()
        second = c.post("/generate", json=body).json()
        stats = c.get("/stats").json()
    assert first["num_cached_tokens"] == 0
    assert second["num_cached_tokens"] == 48  # 3 full blocks of the 59-token prompt
    assert second["text"] == first["text"]
    assert stats["prefix_caching"] is True
    assert stats["prefix_cache_hit_tokens"] == 48
    assert stats["prefix_cache_hit_rate"] == round(48 / 118, 4)
    assert stats["kv_used_tokens"] == 0 and stats["kv_cached_tokens"] > 0
