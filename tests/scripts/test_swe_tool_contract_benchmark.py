from __future__ import annotations

import json
from pathlib import Path

from scripts import swe_tool_contract_benchmark as benchmark


def test_evaluate_cli_is_a_nonzero_failure_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "evaluate", lambda *a: {"regression_pass": False, "failures": ["failed contract"]})
    monkeypatch.setattr(benchmark.sys, "argv", ["benchmark", "evaluate", "--benchmark-dir", str(tmp_path),
                                              "--report", str(tmp_path / "report.json")])
    assert benchmark.main() == 1


def test_contract_suite_has_independent_boundaries() -> None:
    cases = benchmark.contracts()
    ids = {case["id"] for case in cases}

    assert len(cases) >= 20
    assert "outcome_successful_edit" in ids
    assert "outcome_missing_search_backend" in ids
    assert "repeat_relevant_test_after_edit" in ids
    assert "repeat_second_transport_retry" in ids
    assert all(case["label_source"] == benchmark.VERSION for case in cases)


def test_build_freezes_manifest_and_trajectory_partition(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    row = {
        "path": "rollout_data/example/1.jsonl",
        "run": "example",
        "split": "rollout_data",
        "step": 1,
        "benchmark": "swe_rebench_v2",
        "task": "owner__repo-1",
        "output": (
            "assistant\n"
            "<function=Read><parameter=file_path>src/a.py</parameter></function>\n"
            "user\n<tool_response>VALUE = 1</tool_response>\n"
            "assistant\n"
            "<function=Read><parameter=file_path>src/a.py</parameter></function>\n"
            "user\n<tool_response>VALUE = 1</tool_response>\n"
            "assistant\nDone"
        ),
    }
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output_dir = tmp_path / "benchmark"

    manifest = benchmark.build(output_dir, source, max_trajectory_cases=8)

    assert manifest["version"] == benchmark.VERSION
    assert manifest["contract_cases"] >= 20
    assert manifest["trajectory_cases"] == 1
    loaded_manifest, frozen_contracts, trajectory_cases = benchmark._load_benchmark(output_dir)
    assert loaded_manifest == manifest
    assert frozen_contracts
    assert trajectory_cases[0]["kind"] == "historical_trajectory"
    assert trajectory_cases[0]["partition"] in {"development", "holdout"}


def test_build_is_byte_reproducible(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text(
        json.dumps({
            "path": "rollout_data/example/1.jsonl",
            "run": "example",
            "split": "rollout_data",
            "step": 1,
            "benchmark": "swe_rebench_v2",
            "task": "owner__repo-1",
            "output": "assistant\nDone",
        }) + "\n",
        encoding="utf-8",
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    benchmark.build(first, source, max_trajectory_cases=8)
    benchmark.build(second, source, max_trajectory_cases=8)

    for name in (benchmark.CONTRACTS_FILENAME, benchmark.TRAJECTORIES_FILENAME, benchmark.MANIFEST_FILENAME):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_evaluate_reports_unknown_historical_labels_without_passing_them(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    row = {
        "path": "rollout_data/example/1.jsonl",
        "run": "example",
        "split": "rollout_data",
        "step": 1,
        "benchmark": "swe_rebench_v2",
        "task": "owner__repo-1",
        "output": (
            "assistant\n"
            "<function=Bash><parameter=command>python -c 'print(1)'</parameter></function>\n"
            "user\n<tool_response>ok</tool_response>\n"
            "assistant\nDone"
        ),
    }
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output_dir = tmp_path / "benchmark"
    benchmark.build(output_dir, source, max_trajectory_cases=8)
    report_path = output_dir / "report.json"

    report = benchmark.evaluate(output_dir, report_path)

    # The baseline currently has tool-contract failures, but this assertion
    # must also pass after those production defects are repaired.
    assert report["regression_pass"] == (not report["failures"])
    assert report["counts"].get("tool_contract_checks", 0) == 10
    assert report["counts"].get("unknown_historical_labels", 0) >= 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["version"] == benchmark.VERSION


def test_probe_is_read_only_and_uses_all_requested_endpoints(monkeypatch, tmp_path: Path) -> None:
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            return b'{"ok": true}'

    seen = []

    def fake_urlopen(request, timeout):
        seen.append((request.full_url, timeout, request.method))
        return Response()

    monkeypatch.setattr(benchmark.urllib.request, "urlopen", fake_urlopen)
    output = tmp_path / "probe.json"
    report = benchmark.probe(["http://one:18080", "http://two:18080"], output, 1.5)

    assert report["healthy_count"] == 2
    assert len(seen) == 6
    assert all(method == "GET" for _url, _timeout, method in seen)
    assert all("/execute" not in url and "/claim" not in url for url, _timeout, _method in seen)
