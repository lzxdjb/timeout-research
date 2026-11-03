"""Isolated deterministic reasoning validation over synchronized student replicas.

The trainer invokes this while the existing rollout replicas are awake and
after their weights have been synchronized. It never changes ``VAL_FILES`` or
the strict SWE validation roster, and never contacts a teacher endpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import aiohttp

from verl.utils.reward_score.swe_audio_reasoning_opd import compute_general_score

logger = logging.getLogger(__name__)
_BENCHMARKS = frozenset({"aime_2024", "aime_2025", "gpqa_diamond", "mmlu_pro", "bbh"})


def _normalize_api_base(address: str) -> str:
    address = str(address).strip().rstrip("/")
    if not address:
        raise ValueError("Empty reasoning validation rollout endpoint")
    if not address.startswith(("http://", "https://")):
        address = f"http://{address}"
    return address if address.endswith("/v1") else f"{address}/v1"


def _load_rows(config: Mapping[str, Any]) -> list[dict]:
    import pyarrow.parquet as pq

    files = list(config.get("val_files") or [])
    if len(files) != len(_BENCHMARKS) or {Path(str(path)).stem for path in files} != _BENCHMARKS:
        raise ValueError("Reasoning validation requires exactly aime_2024, aime_2025, gpqa_diamond, mmlu_pro, bbh")
    raw_limit = config.get("max_samples")
    limit = -1 if raw_limit is None else int(raw_limit)
    if limit == 0 or limit < -1:
        raise ValueError("Reasoning max_samples must be positive, -1, or null")
    rows = []
    for raw_path in files:
        path = Path(str(raw_path)).expanduser().resolve()
        benchmark = path.stem
        parquet = pq.ParquetFile(path)
        table = parquet.read() if limit < 0 else next(parquet.iter_batches(batch_size=limit), None)
        records = [] if table is None else table.to_pylist()
        if limit > 0:
            records = records[:limit]
        if not records:
            raise ValueError(f"Empty reasoning validation file: {path}")
        for row in records:
            if row.get("data_source") != f"benchmark_{benchmark}":
                raise ValueError(f"Unexpected reasoning source in {path}: {row.get('data_source')!r}")
            prompt = row.get("prompt")
            if not isinstance(prompt, list) or not prompt:
                raise ValueError(f"Missing reasoning validation prompt in {path}")
            if row.get("images") or row.get("audios") or row.get("videos"):
                raise ValueError(f"The selected reasoning validation benchmarks must contain text prompts: {path}")
            reward = row.get("reward_model")
            if not isinstance(reward, dict) or reward.get("ground_truth") is None:
                raise ValueError(f"Missing reasoning validation ground truth in {path}")
            rows.append({**row, "benchmark": benchmark})
    return rows


def _prompt_length(tokenizer, messages: list[dict]) -> int:
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=True)
    if isinstance(tokens, dict):
        tokens = tokens["input_ids"]
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if tokens and isinstance(tokens[0], list):
        if len(tokens) != 1:
            raise ValueError("Reasoning prompt tokenizer returned more than one prompt")
        tokens = tokens[0]
    return len(tokens)


def _metrics(records: list[dict]) -> dict[str, float]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["benchmark"]].append(record)
    metrics = {}
    for name, samples in [*sorted(grouped.items()), ("overall", records)]:
        prefix = f"val-general/{name}"
        metrics[f"{prefix}/count"] = float(len(samples))
        for key in ("score", "acc", "deterministic_acc", "format_valid"):
            metrics[f"{prefix}/{key}"] = sum(float(item["reward"][key]) for item in samples) / len(samples)
        metrics[f"{prefix}/response_tokens_mean"] = sum(item["response_tokens"] for item in samples) / len(samples)
        metrics[f"{prefix}/truncated_fraction"] = sum(item["finish_reason"] == "length" for item in samples) / len(
            samples
        )
    metrics["val-general/completed"] = 1.0
    return metrics


def _write_summary(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


async def _run_requests(config, api_bases, rows, tokenizer, journal) -> list[dict]:
    concurrency = int(config.get("concurrency", 32))
    if concurrency <= 0:
        raise ValueError("Reasoning validation concurrency must be positive")
    response_length = int(config.get("response_length", 253951))
    max_model_len = int(config.get("max_model_len", 262144))
    max_prompt_length = int(config.get("max_prompt_length", 8192))
    timeout = float(config.get("timeout", 3600))
    if min(response_length, max_model_len, max_prompt_length, timeout) <= 0:
        raise ValueError("Reasoning validation token budgets and timeout must be positive")
    headers = {}
    api_key = config.get("api_key")
    if api_key and api_key != "dummy":
        headers["Authorization"] = f"Bearer {api_key}"
    inflight = [0] * len(api_bases)
    next_replica = 0
    semaphore = asyncio.Semaphore(concurrency)
    timeout_config = aiohttp.ClientTimeout(total=timeout, connect=30)
    async with aiohttp.ClientSession(
        timeout=timeout_config, connector=aiohttp.TCPConnector(limit=concurrency), headers=headers, trust_env=False
    ) as session:
        model = config.get("served_model_name")
        if not model:
            async with session.get(f"{api_bases[0]}/models") as response:
                response.raise_for_status()
                discovery = await response.json()
            models = discovery.get("data") or []
            if not models or not isinstance(models[0], dict) or not models[0].get("id"):
                raise RuntimeError("Student reasoning validation endpoint did not return a served model")
            model = str(models[0]["id"])

        async def evaluate(index, row):
            nonlocal next_replica
            async with semaphore:
                # There is no await between choosing and incrementing, so
                # requests select the least busy replica without a race.
                minimum = min(inflight)
                replica = next(
                    (next_replica + offset) % len(api_bases)
                    for offset in range(len(api_bases))
                    if inflight[(next_replica + offset) % len(api_bases)] == minimum
                )
                inflight[replica] += 1
                next_replica = (replica + 1) % len(api_bases)
                try:
                    base = api_bases[replica]
                    messages = row["prompt"]
                    if tokenizer is None:
                        async with session.post(
                            f"{base.removesuffix('/v1')}/tokenize",
                            json={
                                "model": model,
                                "messages": messages,
                                "add_generation_prompt": True,
                                "chat_template_kwargs": {"enable_thinking": True},
                            },
                        ) as response:
                            response.raise_for_status()
                            tokenized = await response.json()
                        prompt_tokens = int(tokenized["count"])
                    else:
                        prompt_tokens = _prompt_length(tokenizer, messages)
                    if prompt_tokens > max_prompt_length:
                        raise ValueError(
                            f"Reasoning validation prompt exceeds {max_prompt_length}: "
                            f"{row.get('uid', index)!r} has {prompt_tokens} tokens"
                        )
                    max_tokens = min(response_length, max_model_len - prompt_tokens)
                    if max_tokens <= 0:
                        raise ValueError(
                            f"Reasoning validation prompt exhausts the model context: {row.get('uid', index)}"
                        )
                    payload = {
                        "model": model,
                        "messages": messages,
                        "n": 1,
                        "stream": False,
                        "max_tokens": max_tokens,
                        "temperature": float(config.get("temperature", 1.0)),
                        "top_p": float(config.get("top_p", 1.0)),
                        "top_k": int(config.get("top_k", -1)),
                        "chat_template_kwargs": {"enable_thinking": True},
                    }
                    async with session.post(f"{base}/chat/completions", json=payload) as response:
                        response.raise_for_status()
                        result = await response.json()
                    choices = result.get("choices") or []
                    if len(choices) != 1 or not isinstance(choices[0].get("message"), dict):
                        raise RuntimeError(f"Malformed reasoning validation completion for {row.get('uid', index)!r}")
                    choice = choices[0]
                    if choice.get("finish_reason") not in {"stop", "length"}:
                        raise RuntimeError(f"Unsuccessful reasoning completion: {choice.get('finish_reason')!r}")
                    message = choice["message"]
                    solution = message.get("content") or ""
                    if not isinstance(solution, str):
                        raise TypeError("Reasoning validation completion content must be text")
                    info = dict(row.get("extra_info") or {})
                    reward_spec = row["reward_model"]
                    if "equivalent_answers" in reward_spec:
                        info.setdefault("equivalent_answers", reward_spec["equivalent_answers"])
                    reward = await asyncio.to_thread(
                        compute_general_score,
                        row["data_source"],
                        solution,
                        reward_spec["ground_truth"],
                        info,
                        reasoning_reward_path=config.get("reasoning_reward_path"),
                    )
                    usage = result.get("usage") or {}
                    record = {
                        "index": index,
                        "uid": row.get("uid", str(index)),
                        "benchmark": row["benchmark"],
                        "data_source": row["data_source"],
                        "student_api_base": base,
                        "prompt": messages,
                        "ground_truth": reward_spec["ground_truth"],
                        "response": solution,
                        "reasoning_content": message.get("reasoning_content", message.get("reasoning")),
                        "reward": reward,
                        "finish_reason": choice["finish_reason"],
                        "prompt_tokens": int(usage.get("prompt_tokens", prompt_tokens)),
                        "response_tokens": int(usage.get("completion_tokens", 0)),
                        "requested_max_tokens": max_tokens,
                    }
                    journal.write(json.dumps(record, ensure_ascii=False) + "\n")
                    journal.flush()
                    return record
                finally:
                    inflight[replica] -= 1

        tasks = [asyncio.create_task(evaluate(index, row)) for index, row in enumerate(rows)]
        try:
            return await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def run_general_reasoning_validation(config, api_bases, step, tokenizer=None) -> dict[str, float]:
    """Run the five selected benchmarks against current student weights.

    ``max_samples`` is a per-file limit: -1/null selects every example. Transport,
    parsing, and scorer errors fail validation instead of becoming zero rewards.
    """
    if not config or not bool(config.get("enabled", False)):
        return {}
    if config.get("failure_policy", "error") != "error":
        raise ValueError("Reasoning validation supports only failure_policy=error")
    bases = list(dict.fromkeys(_normalize_api_base(address) for address in api_bases))
    if not bases:
        raise ValueError("Reasoning validation requires synchronized student rollout endpoints")
    rows = _load_rows(config)
    output_root = Path(str(config["output_dir"])).expanduser().resolve()
    run_id = f"step-{int(step)}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:12]}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    summary_path = run_dir / "summary.json"
    summary = {
        "run_id": run_id,
        "step": int(step),
        "status": "running",
        "samples": len(rows),
        "student_api_bases": bases,
        "val_files": [str(path) for path in config["val_files"]],
    }
    _write_summary(summary_path, summary)
    logger.info("General reasoning validation: step=%s samples=%s output=%s", step, len(rows), run_dir)
    try:
        with (run_dir / "samples.jsonl").open("x", encoding="utf-8") as journal:
            records = asyncio.run(_run_requests(config, bases, rows, tokenizer, journal))
        metrics = _metrics(records)
    except Exception as error:
        _write_summary(summary_path, {**summary, "status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    _write_summary(summary_path, {**summary, "status": "completed", "metrics": metrics})
    return metrics
