"""HTTP API: ``POST /generate`` streams tokens over Server-Sent Events, ``GET /stats`` reports load.

Run it with::

    tinyserve-server --strategy continuous --kv-layout paged

and stream a completion with::

    curl -N -X POST localhost:8000/generate \\
         -H 'Content-Type: application/json' \\
         -d '{"prompt": "Hello", "max_tokens": 32}'
"""

from __future__ import annotations

import argparse
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from tinyserve.async_engine import AsyncEngine
from tinyserve.engine import KV_LAYOUTS, STRATEGIES, EngineConfig, StepOutput, build_engine
from tinyserve.kv_quant import KV_DTYPES
from tinyserve.request import SamplingParams
from tinyserve.tokenizer import CharTokenizer


class GenerateRequest(BaseModel):
    """Body of ``POST /generate``."""

    prompt: str = Field(min_length=1, description="Input text (printable ASCII).")
    max_tokens: int = Field(64, ge=1, description="Number of tokens to generate at most.")
    temperature: float = Field(0.0, ge=0.0, description="0 = greedy.")
    ignore_eos: bool = Field(False, description="Keep generating past end-of-sequence.")
    seed: int | None = Field(None, description="Seed for sampling, for reproducible output.")
    stream: bool = Field(True, description="Stream tokens as SSE, or return one JSON body.")


def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _summary(out: StepOutput, text: str) -> dict[str, Any]:
    req = out.request
    return {
        "finished": True,
        "text": text,
        "finish_reason": req.finish_reason.value if req.finish_reason else None,
        "num_prompt_tokens": len(req.prompt_ids),
        "num_cached_tokens": req.num_cached_tokens,
        "num_generated_tokens": len(req.output_ids),
        "ttft_ms": round(req.ttft * 1000, 2) if req.ttft is not None else None,
        "latency_ms": round(req.latency * 1000, 2) if req.latency is not None else None,
        "num_preemptions": req.num_preemptions,
    }


def create_app(config: EngineConfig | None = None) -> FastAPI:
    """Build the FastAPI app around an engine described by ``config``."""
    config = config or EngineConfig(
        strategy="continuous", kv_layout="paged", enable_prefix_caching=True
    )
    tokenizer = CharTokenizer()
    async_engine = AsyncEngine(build_engine(config))

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        async_engine.start()
        yield
        async_engine.stop()

    app = FastAPI(
        title="TinyServe",
        description="A tiny LLM inference engine (random weights: the output is gibberish).",
        lifespan=lifespan,
    )
    app.state.engine = async_engine

    @app.post("/generate")
    async def generate(body: GenerateRequest) -> Any:
        """Generate tokens for ``body.prompt``, streamed as SSE unless ``stream`` is false."""
        params = SamplingParams(
            max_new_tokens=body.max_tokens,
            temperature=body.temperature,
            ignore_eos=body.ignore_eos,
            seed=body.seed,
        )
        tokens = async_engine.generate(tokenizer.encode(body.prompt), params)
        try:
            first = await anext(tokens)  # surfaces validation errors as HTTP 400
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err

        if not body.stream:
            out, text = first, tokenizer.decode_token(first.token_id)
            async for out in tokens:
                text += tokenizer.decode_token(out.token_id)
            return _summary(out, text)

        async def event_stream() -> AsyncIterator[str]:
            text, index, out = "", 0, first
            try:
                while True:
                    piece = tokenizer.decode_token(out.token_id)
                    text += piece
                    yield _sse({"index": index, "token": piece, "token_id": out.token_id})
                    index += 1
                    if out.finished:
                        yield _sse(_summary(out, text))
                        return
                    out = await anext(tokens)
            finally:
                await tokens.aclose()  # client went away: abort and free its KV memory

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/stats")
    async def stats() -> dict[str, Any]:
        """Queue depth, batch size, KV memory, prefix-cache hit rate and recent tokens/sec."""
        return async_engine.stats()

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    return app


def main() -> None:
    """Command-line entry point: ``tinyserve-server``."""
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the TinyServe HTTP server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--strategy", choices=STRATEGIES, default="continuous")
    parser.add_argument("--kv-layout", choices=KV_LAYOUTS, default="paged")
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--kv-capacity-tokens", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-dtype", choices=KV_DTYPES, default="float32")
    parser.add_argument(
        "--kv-memory-mib",
        type=float,
        default=None,
        help="size the KV pool in MiB instead of tokens (smaller dtypes then fit more)",
    )
    parser.add_argument(
        "--prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="share KV blocks of common prompt prefixes (paged layout only)",
    )
    parser.add_argument("--max-tokens-per-step", type=int, default=2048)
    parser.add_argument(
        "--chunked-prefill", action="store_true", help="split long prefills across steps"
    )
    parser.add_argument("--threads", type=int, default=None, help="PyTorch CPU threads")
    args = parser.parse_args()
    if args.strategy != "continuous" and args.kv_layout == "paged":
        args.kv_layout = "contiguous"  # paged needs preemption, which only continuous has
    if args.kv_layout != "paged":
        args.prefix_caching = False

    config = EngineConfig(
        strategy=args.strategy,
        kv_capacity_tokens=args.kv_capacity_tokens,
        max_batch_size=args.max_batch_size,
        kv_layout=args.kv_layout,
        block_size=args.block_size,
        enable_prefix_caching=args.prefix_caching,
        max_tokens_per_step=args.max_tokens_per_step,
        chunked_prefill=args.chunked_prefill,
        num_threads=args.threads,
        kv_dtype=args.kv_dtype,
        kv_memory_bytes=None if args.kv_memory_mib is None else int(args.kv_memory_mib * 2**20),
    )
    uvicorn.run(create_app(config), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
