#!/usr/bin/env python3
"""Keep a vLLM judge busy with concurrent dummy requests; stop with Ctrl+C.

    python scripts/load_swe_repeat_judge.py --concurrency 16 --max-tokens 1024
    python scripts/load_swe_repeat_judge.py --duration 60 --prompt-chars 32768

Uses SWE_AGENT_REPEAT_JUDGE_URL and SWE_AGENT_REPEAT_JUDGE_MODEL. Requires aiohttp.
"""

import argparse
import asyncio
import math
import os
import time
import uuid
from pathlib import Path

import aiohttp


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.environ.get("SWE_AGENT_REPEAT_JUDGE_URL"))
    parser.add_argument("--model", default=os.environ.get("SWE_AGENT_REPEAT_JUDGE_MODEL", "swe-repeat-judge"))
    parser.add_argument("--concurrency", type=int, default=16, help="simultaneous requests (default: 16)")
    parser.add_argument("--prompt-chars", type=int, default=16384, help="dummy trace length in characters (default: 16384)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="generated tokens per request (default: 1024)")
    parser.add_argument("--duration", type=float, default=0, help="run for this many seconds; 0 means until Ctrl+C")
    parser.add_argument("--timeout", type=float, default=300, help="timeout per request in seconds (default: 300)")
    parser.add_argument("--log-interval", type=float, default=5, help="progress interval in seconds (default: 5)")
    args = parser.parse_args()
    if not args.url:
        parser.error("set SWE_AGENT_REPEAT_JUDGE_URL or pass --url")
    for name in ("concurrency", "prompt_chars", "max_tokens", "timeout", "log_interval"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error("--duration must be nonnegative")
    return args


async def run(args):
    base_url = args.url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    url = base_url + "/chat/completions"
    headers = {}
    key_file = os.environ.get("SWE_AGENT_REPEAT_JUDGE_API_KEY_FILE")
    if key_file:
        headers["Authorization"] = "Bearer " + Path(key_file).read_text().strip()

    trace_line = (
        "Tool: Read src/example.py. Observation: found a function that validates input. "
        "Tool: Edit src/example.py. Observation: added a boundary check. "
        "Tool: Bash pytest tests/test_example.py. Observation: tests passed.\n"
    )
    trace = (trace_line * (args.prompt_chars // len(trace_line) + 1))[:args.prompt_chars]
    stats = dict(ok=0, errors=0, inflight=0, prompt_tokens=0, output_tokens=0, latency=0.0)
    started = time.monotonic()
    last_error = 0.0

    def report():
        elapsed = max(time.monotonic() - started, 0.001)
        print(
            f"elapsed={elapsed:.0f}s ok={stats['ok']} errors={stats['errors']} "
            f"inflight={stats['inflight']} "
            f"avg_input_tok/s={stats['prompt_tokens'] / elapsed:.1f} "
            f"avg_output_tok/s={stats['output_tokens'] / elapsed:.1f} "
            f"avg_latency={stats['latency'] / max(stats['ok'], 1):.2f}s",
            flush=True,
        )

    async def worker(session):
        nonlocal last_error
        while True:
            # A unique prefix makes each request perform prefill instead of reusing the whole prompt cache.
            prompt = (
                f"Dummy request {uuid.uuid4().hex}\n{trace}\n"
                "Write a long, detailed, numbered review of this synthetic trace. Continue until the token limit."
            )
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                "temperature": 0.8,
                "stream": False,
                "ignore_eos": True,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            request_started = time.monotonic()
            stats["inflight"] += 1
            failed = False
            try:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:240]}")
                    reply = await response.json()
                if not isinstance(reply, dict) or not reply.get("choices"):
                    raise ValueError(f"Completion response has no choices: {str(reply)[:240]}")
                usage = reply.get("usage") or {}
                if not isinstance(usage, dict):
                    raise ValueError("Completion response has invalid usage metadata")
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                output_tokens = int(usage.get("completion_tokens") or 0)
                stats["prompt_tokens"] += prompt_tokens
                stats["output_tokens"] += output_tokens
                stats["ok"] += 1
                stats["latency"] += time.monotonic() - request_started
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, TypeError, RuntimeError) as exc:
                failed = True
                stats["errors"] += 1
                now = time.monotonic()
                if stats["errors"] == 1 or now - last_error >= 5:
                    print(f"Request error: {type(exc).__name__}: {exc}", flush=True)
                    last_error = now
            finally:
                stats["inflight"] -= 1
            if failed:
                await asyncio.sleep(1)

    async def reporter():
        while True:
            await asyncio.sleep(args.log_interval)
            report()

    print(
        f"Loading {url} model={args.model} concurrency={args.concurrency} "
        f"prompt_chars={args.prompt_chars} max_tokens={args.max_tokens} "
        f"duration={'until Ctrl+C' if not args.duration else str(args.duration) + 's'}",
        flush=True,
    )
    # Direct pod requests must bypass HTTP_PROXY/HTTPS_PROXY.
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=args.timeout),
        connector=aiohttp.TCPConnector(limit=args.concurrency),
        headers=headers,
        trust_env=False,
    ) as session:
        tasks = [asyncio.create_task(worker(session)) for _ in range(args.concurrency)]
        tasks.append(asyncio.create_task(reporter()))
        try:
            if args.duration:
                await asyncio.sleep(args.duration)
            else:
                await asyncio.Event().wait()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            report()


if __name__ == "__main__":
    try:
        asyncio.run(run(parse_args()))
    except KeyboardInterrupt:
        print("Stopped.", flush=True)
