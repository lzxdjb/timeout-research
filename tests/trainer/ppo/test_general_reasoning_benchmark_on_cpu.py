"""End-to-end HTTP validation against local student replicas, without GPUs."""

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import aiohttp
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from verl.trainer.ppo import general_reasoning_benchmark as benchmark


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": True, "enable_thinking": True}
        return [1] * 100


@contextmanager
def _student_server(calls, *, fail=False):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            assert self.path == "/v1/models"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": "current-student"}]}).encode())

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                assert payload["chat_template_kwargs"] == {"enable_thinking": True}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"count":100}')
                return
            assert self.path == "/v1/chat/completions"
            calls.append(payload)
            self.send_response(500 if fail else 200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if fail:
                self.wfile.write(b'{"error":"test student failure"}')
                return
            prompt = payload["messages"][-1]["content"]
            solution = r"\boxed{B}" if prompt.endswith("-0") else "The answer is B"
            self.wfile.write(
                json.dumps(
                    {
                        "choices": [{"message": {"content": solution}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 8},
                    }
                ).encode()
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _config(tmp_path, limit=-1):
    files = []
    for name in ("aime_2024", "aime_2025", "gpqa_diamond", "mmlu_pro", "bbh"):
        path = tmp_path / f"{name}.parquet"
        rows = [
            {
                "uid": f"{name}-{index}",
                "data_source": f"benchmark_{name}",
                "prompt": [{"role": "user", "content": f"{name}-{index}"}],
                "reward_model": {"ground_truth": "B"},
                "extra_info": {"answer_type": "multiple_choice"},
            }
            for index in range(2)
        ]
        pq.write_table(pa.Table.from_pylist(rows), path)
        files.append(str(path))
    return {
        "enabled": True,
        "val_files": files,
        "max_samples": limit,
        "concurrency": 4,
        "output_dir": str(tmp_path / "out"),
    }


def test_http_validation_uses_student_replicas_and_separate_binary_metrics(tmp_path):
    config = _config(tmp_path)
    calls1, calls2 = [], []
    with _student_server(calls1) as base1, _student_server(calls2) as base2:
        metrics = benchmark.run_general_reasoning_validation(config, [base1, base2], 7, tokenizer=Tokenizer())
    assert calls1 and calls2
    assert len(calls1) + len(calls2) == 10
    for request in calls1 + calls2:
        assert request["model"] == "current-student"
        assert request["max_tokens"] == 253951
        assert request["n"] == 1 and request["chat_template_kwargs"] == {"enable_thinking": True}
    assert metrics["val-general/overall/acc"] == 0.5
    assert metrics["val-general/overall/deterministic_acc"] == 1.0
    assert metrics["val-general/overall/count"] == 10.0
    for name in benchmark._BENCHMARKS:
        assert metrics[f"val-general/{name}/acc"] == 0.5
    output = next((tmp_path / "out").iterdir())
    summary = json.loads((output / "summary.json").read_text())
    assert summary["step"] == 7 and summary["status"] == "completed"
    records = [json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()]
    assert len(records) == 10 and {row["reward"]["score"] for row in records} == {0.0, 1.0}


def test_smoke_limit_is_per_dataset(tmp_path):
    config = _config(tmp_path, limit=1)
    assert len(benchmark._load_rows(config)) == 5


def test_response_budget_reserves_the_rendered_prompt_context(tmp_path):
    config = _config(tmp_path, limit=1)
    config["max_model_len"] = 140
    config["response_length"] = 100
    calls = []
    with _student_server(calls) as base:
        benchmark.run_general_reasoning_validation(config, [base], 1, tokenizer=Tokenizer())
    assert len(calls) == 5
    assert {request["max_tokens"] for request in calls} == {40}


def test_server_tokenization_fallback_uses_root_tokenize_endpoint(tmp_path):
    config = _config(tmp_path, limit=1)
    calls = []
    with _student_server(calls) as base:
        metrics = benchmark.run_general_reasoning_validation(config, [base], 1)
    assert metrics["val-general/overall/count"] == 5.0
    assert len(calls) == 5


def test_endpoint_failure_propagates_and_leaves_failed_summary(tmp_path):
    config = _config(tmp_path, limit=1)
    with _student_server([], fail=True) as base:
        with pytest.raises(aiohttp.ClientResponseError):
            benchmark.run_general_reasoning_validation(config, [base], 1, tokenizer=Tokenizer())
    summary = json.loads(next((tmp_path / "out").glob("*/summary.json")).read_text())
    assert summary["status"] == "failed"
    assert "ClientResponseError" in summary["error"]


def test_hle_and_missing_benchmarks_are_rejected(tmp_path):
    config = _config(tmp_path)
    config["val_files"][-1] = str(tmp_path / "hle.parquet")
    with pytest.raises(ValueError, match="requires exactly"):
        benchmark._load_rows(config)


def test_prompt_budget_is_checked_before_generation(tmp_path):
    config = _config(tmp_path, limit=1)
    config["max_prompt_length"] = 10
    calls = []
    with _student_server(calls) as base:
        with pytest.raises(ValueError, match="exceeds 10"):
            benchmark.run_general_reasoning_validation(config, [base], 1, tokenizer=Tokenizer())
    assert calls == []


def test_disabled_profile_does_not_read_files_or_require_endpoints():
    assert benchmark.run_general_reasoning_validation({"enabled": False}, [], 1) == {}
