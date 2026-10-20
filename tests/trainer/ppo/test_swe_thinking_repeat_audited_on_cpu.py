"""Replay the full assistant-reviewed trajectory texts; no excerpt-only tests.

These labels establish regression behavior on selected cases, not population
accuracy. Historical decoded dumps do not prove native rollout alignment.
"""

import concurrent.futures
import gzip
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from verl.utils.swe_thinking_repeat import Config, capture_generation, detect, verified_event

os.environ["TOKENIZERS_PARALLELISM"] = "false"
ROOT = Path(__file__).resolve().parents[3]
FIXTURE = Path(__file__).parent / "fixtures/swe_thinking_repeat_audited_20261020.jsonl.gz"


def replay(case):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(ROOT.parent / "stock-rl-reflect/data/3552-checkpoint/tokenizer.json"))
    assert hashlib.sha256(case["text"].encode()).hexdigest() == case["text_sha256"]
    old = detect(case["text"], tokenizer)
    new = detect(case["text"], tokenizer, Config(aggregate_enabled=True))
    ids = tokenizer.encode("<think>" + case["text"] + "</think>", add_special_tokens=False).ids
    data = SimpleNamespace(response_ids=ids, response_mask=[1] * len(ids), prompt_ids=[42] + ids)
    proxy = SimpleNamespace(
        backend_tokenizer=tokenizer, decode=lambda ids, **kwargs: tokenizer.decode(ids, skip_special_tokens=False)
    )
    event, status = capture_generation(proxy, data, config=Config(aggregate_enabled=True))
    if event:
        assert verified_event(event, ids, data.response_mask), case["id"]
        assert case["repetitive"], case["id"]
    return dict(
        native=bool(event),
        native_status=status,
        id=case["id"],
        kind=case["kind"],
        repetitive=case["repetitive"],
        baseline=old["matched"],
        extended=new["matched"],
        tokens=new["tokens"],
        selected_tokens=new["penalized_tokens"],
        aggregate=any(e["rule"] == "exact_sentence_aggregate" for e in new["evidence"]),
    )


def test_complete_audited_case_set():
    with gzip.open(FIXTURE, "rt") as stream:
        cases = [json.loads(line) for line in stream]
    assert len(cases) == 99
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(replay, cases))
    negatives = [r for r in results if not r["repetitive"]]
    old_hits = [r for r in results if r["kind"] == "current_hit"]
    misses = [r for r in results if r["kind"] == "severe_miss"]
    assert len(negatives) == 18 and not any(r["extended"] for r in negatives)
    assert len(old_hits) == 45 and all(r["baseline"] and r["extended"] for r in old_hits)
    assert sum(r["extended"] for r in misses) >= 11
    assert sum(r["native"] for r in misses) >= 11
    output = ROOT / "analysis/thinking_repeat_runs_20261020/implemented_audit_results.json"
    output.write_text(json.dumps(results, indent=2))
