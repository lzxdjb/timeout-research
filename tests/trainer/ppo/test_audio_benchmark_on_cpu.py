# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from verl.trainer.ppo import audio_benchmark


def _write_summary(
    path: Path,
    *,
    run_id: str,
    status: str = "completed",
    benchmark_status: str = "completed",
    benchmark_id: str = "aishell2",
    metrics: list[dict] | None = None,
) -> None:
    child_summary = path.parent / benchmark_id / "summary.json"
    child_summary.parent.mkdir(parents=True, exist_ok=True)
    child_summary.write_text(
        json.dumps(
            {
                "metrics": metrics
                if metrics is not None
                else [
                    {
                        "metric_id": "aishell2_cer",
                        "raw_values": {"cer": 12.5},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": status,
                "benchmarks": [
                    {
                        "benchmark_id": benchmark_id,
                        "status": benchmark_status,
                        "summary": {"summary_path": str(child_summary)},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_run_audio_benchmarks_reads_fresh_unique_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    gage_root = tmp_path / "gage"
    runner = gage_root / "scripts" / "run" / "run_suite.sh"
    runner.parent.mkdir(parents=True)
    runner.touch()
    output_root = tmp_path / "results"

    stale_summary = output_root / "experiment-step-0" / "suite_summary.json"
    stale_summary.parent.mkdir(parents=True)
    _write_summary(
        stale_summary,
        run_id="experiment-step-0",
        status="completed_with_failures",
        benchmark_status="failed",
    )

    monkeypatch.setattr(audio_benchmark, "uuid4", lambda: SimpleNamespace(hex="0123456789abcdef"))
    launched_run_ids: list[str] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        run_id = command[command.index("--run-id") + 1]
        launched_run_ids.append(run_id)
        assert run_id != "experiment-step-0"
        assert not (output_root / run_id).exists()
        summary = output_root / run_id / "suite_summary.json"
        summary.parent.mkdir(parents=True)
        _write_summary(summary, run_id=run_id)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(audio_benchmark.subprocess, "run", fake_run)
    metrics = audio_benchmark.run_audio_benchmarks(
        {
            "enabled": True,
            "gage_root": str(gage_root),
            "data_root": str(tmp_path / "data"),
            "output_dir": str(output_root),
            "served_model_name": "model",
            "benchmarks": ["aishell2"],
        },
        server_addresses=["127.0.0.1:8000"],
        model_path="/models/test",
        experiment_name="experiment",
        global_step=0,
    )

    assert len(launched_run_ids) == 1
    assert launched_run_ids[0].startswith("experiment-step-0-")
    assert launched_run_ids[0].endswith("-0123456789ab")
    assert metrics["val-audio/aishell2/aishell2_cer/cer"] == 12.5
    assert metrics["val-audio/completed"] == 1.0


@pytest.mark.parametrize("include_additional_audio", [False, True])
def test_supported_benchmarks_use_their_gage_flags_and_dataset_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, include_additional_audio: bool
) -> None:
    gage_root = tmp_path / "gage"
    runner = gage_root / "scripts" / "run" / "run_suite.sh"
    runner.parent.mkdir(parents=True)
    runner.touch()
    data_root = tmp_path / "data"
    output_root = tmp_path / "results"
    expected = {
        "--aishell-2": "aishell2",
        "--chartqa": "chartqa",
        "--clothoaqa": "clothoaqa",
        "--mmbench": "mmbench",
        "--mmstar": "mmstar",
        "--ocrbench": "ocrbench",
        "--realworldqa": "realworldqa",
        "--rul-muchomusic": "muchomusic",
        "--voicebenchbbh": "voicebench_bbh",
    }
    additional_audio = {
        "--aishell-1": "aishell1",
        "--librispeech": "librispeech",
        "--wenetspeech": "wenetspeech",
    }
    if include_additional_audio:
        expected.update(additional_audio)

    monkeypatch.setattr(audio_benchmark, "uuid4", lambda: SimpleNamespace(hex="0123456789abcdef"))

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        for flag, directory in expected.items():
            assert command[command.index(flag) + 1] == str(data_root / directory)
        if not include_additional_audio:
            assert not set(additional_audio).intersection(command)
        run_id = command[command.index("--run-id") + 1]
        summary = output_root / run_id / "suite_summary.json"
        summary.parent.mkdir(parents=True)
        _write_summary(summary, run_id=run_id)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(audio_benchmark.subprocess, "run", fake_run)
    metrics = audio_benchmark.run_audio_benchmarks(
        {
            "enabled": True,
            "gage_root": str(gage_root),
            "data_root": str(data_root),
            "output_dir": str(output_root),
            "served_model_name": "model",
            "max_samples": 20,
            "benchmarks": list(audio_benchmark._BENCHMARK_SPECS) if include_additional_audio else None,
        },
        server_addresses=["127.0.0.1:8000"],
        model_path="/models/test",
        experiment_name="experiment",
        global_step=0,
    )

    assert metrics["val-audio/completed"] == 1.0


def test_full_audio_suite_runs_all_samples_without_vision_benchmarks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gage_root = tmp_path / "gage"
    runner = gage_root / "scripts" / "run" / "run_suite.sh"
    runner.parent.mkdir(parents=True)
    runner.touch()
    data_root = tmp_path / "data"
    output_root = tmp_path / "results"
    benchmarks = ["aishell1", "librispeech", "wenetspeech", "clothoaqa", "rul_muchomusic", "voicebench_bbh"]
    expected_dataset_args = [
        "--aishell-1",
        str(data_root / "aishell1"),
        "--librispeech",
        str(data_root / "librispeech"),
        "--wenetspeech",
        str(data_root / "wenetspeech"),
        "--clothoaqa",
        str(data_root / "clothoaqa"),
        "--rul-muchomusic",
        str(data_root / "muchomusic"),
        "--voicebenchbbh",
        str(data_root / "voicebench_bbh"),
    ]

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        # GAGE interprets an integer after each dataset path as a sample limit.
        # Requiring the exact argument tail also excludes extra dataset flags.
        assert command[command.index("--no-resume") + 1 :] == expected_dataset_args
        assert "--max-samples" not in command
        assert "--sample-seed" not in command
        run_id = command[command.index("--run-id") + 1]
        summary = output_root / run_id / "suite_summary.json"
        summary.parent.mkdir(parents=True)
        _write_summary(summary, run_id=run_id)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(audio_benchmark.subprocess, "run", fake_run)
    metrics = audio_benchmark.run_audio_benchmarks(
        {
            "enabled": True,
            "gage_root": str(gage_root),
            "data_root": str(data_root),
            "output_dir": str(output_root),
            "served_model_name": "model",
            "benchmarks": benchmarks,
            "max_samples": None,
        },
        server_addresses=["127.0.0.1:8000"],
        model_path="/models/test",
        experiment_name="full-audio",
        global_step=0,
    )

    assert metrics["val-audio/completed"] == 1.0


def test_read_metrics_rejects_run_id_mismatch(tmp_path: Path) -> None:
    summary = tmp_path / "suite_summary.json"
    _write_summary(summary, run_id="actual")

    with pytest.raises(RuntimeError, match="run_id mismatch.*expected 'expected'.*found 'actual'"):
        audio_benchmark._read_metrics(summary, expected_run_id="expected")


def test_read_metrics_preserves_genuine_suite_failure(tmp_path: Path) -> None:
    summary = tmp_path / "suite_summary.json"
    _write_summary(
        summary,
        run_id="failed-run",
        status="completed_with_failures",
        benchmark_status="failed",
    )

    with pytest.raises(RuntimeError, match="completed_with_failures.*failed benchmarks: aishell2"):
        audio_benchmark._read_metrics(summary, expected_run_id="failed-run")


@pytest.mark.parametrize(
    ("benchmark_id", "metric_id", "name", "tasks", "scores"),
    [
        (
            "librispeech",
            "librispeech_wer",
            "wer",
            ("librispeech_test_clean_full_eval", "librispeech_test_other_full_eval"),
            (8.518518518518519, 0.0),
        ),
        (
            "wenetspeech",
            "wenetspeech_cer",
            "cer",
            ("wenetspeech_test_net_full_eval", "wenetspeech_test_meeting_full_eval"),
            (0.0, 33.65079365079365),
        ),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_read_metrics_keeps_both_asr_splits_without_an_ambiguous_alias(
    tmp_path: Path, caplog, benchmark_id, metric_id, name, tasks, scores, reverse
) -> None:
    rows = [
        {
            "metric_id": metric_id,
            "task_id": task,
            "count": count,
            "raw_values": {name: score},
            "values": {name: f"{score:.5f}"},
        }
        for task, score, count in zip(tasks, scores, (3, 2), strict=True)
    ]
    if reverse:
        rows.reverse()
    summary = tmp_path / "suite_summary.json"
    _write_summary(summary, run_id="split-run", benchmark_id=benchmark_id, metrics=rows)

    result = audio_benchmark._read_metrics(summary, expected_run_id="split-run")

    assert result == {
        **{
            f"val-audio/{benchmark_id}/{task}/{metric_id}/{name}": score
            for task, score in zip(tasks, scores, strict=True)
        },
        "val-audio/completed": 1.0,
    }
    legacy_key = f"val-audio/{benchmark_id}/{metric_id}/{name}"
    assert f"Omitting ambiguous GAGE audio metric key {legacy_key}" in caplog.text


def test_read_metrics_keeps_single_task_alias_and_existing_value_parsing(tmp_path: Path) -> None:
    summary = tmp_path / "suite_summary.json"
    _write_summary(
        summary,
        run_id="single-run",
        metrics=[
            {
                "metric_id": "aishell2_cer",
                "task_id": "aishell2_test",
                "values": {"cer": "12.50000", "enabled": True, "description": "not a score"},
            }
        ],
    )

    assert audio_benchmark._read_metrics(summary) == {
        "val-audio/aishell2/aishell2_test/aishell2_cer/cer": 12.5,
        "val-audio/aishell2/aishell2_cer/cer": 12.5,
        "val-audio/completed": 1.0,
    }


def test_read_metrics_keeps_distinct_metric_aliases_across_tasks(tmp_path: Path) -> None:
    summary = tmp_path / "suite_summary.json"
    _write_summary(
        summary,
        run_id="clotho-run",
        benchmark_id="clothoaqa",
        metrics=[
            {
                "metric_id": f"clothoaqa_{split}_filtered_accuracy",
                "task_id": f"clothoaqa_{split}_filtered_full_eval",
                "raw_values": {"accuracy": score},
            }
            for split, score in (("val", 0.5), ("test", 1.0))
        ],
    )

    assert audio_benchmark._read_metrics(summary) == {
        "val-audio/clothoaqa/clothoaqa_val_filtered_full_eval/clothoaqa_val_filtered_accuracy/accuracy": 0.5,
        "val-audio/clothoaqa/clothoaqa_test_filtered_full_eval/clothoaqa_test_filtered_accuracy/accuracy": 1.0,
        "val-audio/clothoaqa/clothoaqa_val_filtered_accuracy/accuracy": 0.5,
        "val-audio/clothoaqa/clothoaqa_test_filtered_accuracy/accuracy": 1.0,
        "val-audio/completed": 1.0,
    }


@pytest.mark.parametrize("task_id", [None, "duplicate_task"])
def test_read_metrics_warns_and_omits_duplicate_keys_instead_of_overwriting(tmp_path: Path, caplog, task_id) -> None:
    summary = tmp_path / "suite_summary.json"
    _write_summary(
        summary,
        run_id="duplicate-run",
        metrics=[
            {"metric_id": "aishell2_cer", "task_id": task_id, "raw_values": {"cer": value}}
            for value in (12.5, 20.0)
        ],
    )

    assert audio_benchmark._read_metrics(summary) == {"val-audio/completed": 1.0}
    assert "Omitting ambiguous GAGE audio metric key val-audio/aishell2/aishell2_cer/cer" in caplog.text


def test_success_without_expected_summary_reports_recent_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gage_root = tmp_path / "gage"
    runner = gage_root / "scripts" / "run" / "run_suite.sh"
    runner.parent.mkdir(parents=True)
    runner.touch()
    output_root = tmp_path / "results"
    sibling = output_root / "experiment-step-0-2"
    sibling.mkdir(parents=True)

    monkeypatch.setattr(audio_benchmark, "uuid4", lambda: SimpleNamespace(hex="fedcba9876543210"))
    monkeypatch.setattr(
        audio_benchmark.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
    )

    with pytest.raises(FileNotFoundError, match="did not write the expected summary.*experiment-step-0-2"):
        audio_benchmark.run_audio_benchmarks(
            {
                "enabled": True,
                "gage_root": str(gage_root),
                "data_root": str(tmp_path / "data"),
                "output_dir": str(output_root),
                "served_model_name": "model",
                "benchmarks": ["aishell2"],
            },
            server_addresses=["127.0.0.1:8000"],
            model_path="/models/test",
            experiment_name="experiment",
            global_step=0,
        )


@pytest.mark.parametrize(
    ("addresses", "overrides", "expected_bases", "balanced"),
    [
        (
            ["127.0.0.1:8000", "http://127.0.0.1:8001/v1", "http://127.0.0.1:8000/v1/"],
            {},
            ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"],
            True,
        ),
        (["127.0.0.1:8000"], {}, ["http://127.0.0.1:8000/v1"], False),
        (
            ["127.0.0.1:8000", "http://127.0.0.1:8000/v1/"],
            {},
            ["http://127.0.0.1:8000/v1"],
            False,
        ),
        (
            ["127.0.0.1:8000", "127.0.0.1:8001"],
            {"load_balance": False},
            ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"],
            False,
        ),
        (
            ["127.0.0.1:8000", "127.0.0.1:8001"],
            {"api_base": "http://dedicated:9000/v1"},
            ["http://dedicated:9000/v1"],
            False,
        ),
    ],
)
def test_audio_replica_routing_preserves_endpoint_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, addresses, overrides, expected_bases, balanced
) -> None:
    runner = tmp_path / "gage/scripts/run/run_suite.sh"
    runner.parent.mkdir(parents=True)
    runner.touch()
    output_root = tmp_path / "results"
    router_events = []

    class FakeRouter:
        api_base = "http://127.0.0.1:9001/v1"

        def __init__(self, upstream_api_bases):
            assert balanced
            assert upstream_api_bases == expected_bases

        def __enter__(self):
            router_events.append("started")
            return self

        def __exit__(self, *exc):
            router_events.append("closed")

    def discover_model(api_base, api_key):
        # Discover the model before starting the router, using a real replica.
        assert api_base == expected_bases[0]
        assert api_key == "dummy"
        assert not router_events
        return "trained-audio-model"

    def fake_run(command, **kwargs):
        assert router_events == (["started"] if balanced else [])
        assert command[command.index("--external-api-base") + 1] == (
            FakeRouter.api_base if balanced else expected_bases[0]
        )
        assert command[command.index("--external-model-name") + 1] == "trained-audio-model"
        assert command[command.index("--concurrency") + 1] == "64"
        run_id = command[command.index("--run-id") + 1]
        summary = output_root / run_id / "suite_summary.json"
        summary.parent.mkdir(parents=True)
        _write_summary(summary, run_id=run_id)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(audio_benchmark, "AudioBenchmarkRouter", FakeRouter)
    monkeypatch.setattr(audio_benchmark, "_discover_model", discover_model)
    monkeypatch.setattr(audio_benchmark.subprocess, "run", fake_run)
    metrics = audio_benchmark.run_audio_benchmarks(
        {
            "enabled": True,
            "gage_root": str(tmp_path / "gage"),
            "data_root": str(tmp_path / "data"),
            "output_dir": str(output_root),
            "benchmarks": ["aishell2"],
            "concurrency": 64,
            **overrides,
        },
        server_addresses=addresses,
        model_path="/models/trained-audio-model",
        experiment_name="audio",
        global_step=0,
    )

    assert metrics["val-audio/completed"] == 1.0
    assert router_events == (["started", "closed"] if balanced else [])


@pytest.mark.parametrize("runner_raises", [False, True])
def test_audio_replica_router_closes_when_suite_fails(monkeypatch, tmp_path, runner_raises) -> None:
    runner = tmp_path / "gage/scripts/run/run_suite.sh"
    runner.parent.mkdir(parents=True)
    runner.touch()
    router_events = []

    class FakeRouter:
        api_base = "http://127.0.0.1:9001/v1"

        def __init__(self, upstream_api_bases):
            pass

        def __enter__(self):
            router_events.append("started")
            return self

        def __exit__(self, *exc):
            router_events.append("closed")

    def fake_run(command, **kwargs):
        if runner_raises:
            raise OSError("Cannot launch suite")
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(audio_benchmark, "AudioBenchmarkRouter", FakeRouter)
    monkeypatch.setattr(audio_benchmark.subprocess, "run", fake_run)
    with pytest.raises(OSError if runner_raises else RuntimeError):
        audio_benchmark.run_audio_benchmarks(
            {
                "enabled": True,
                "gage_root": str(tmp_path / "gage"),
                "data_root": str(tmp_path / "data"),
                "output_dir": str(tmp_path / "results"),
                "served_model_name": "model",
                "benchmarks": ["aishell2"],
            },
            server_addresses=["127.0.0.1:8000", "127.0.0.1:8001"],
            model_path="/models/test",
            experiment_name="audio",
            global_step=0,
        )

    assert router_events == ["started", "closed"]


@pytest.mark.parametrize("outcome", ["success", "failure", "exception"])
@pytest.mark.parametrize("explicit_control", [False, True])
def test_audio_control_storage_is_scoped_and_keeps_artifacts_in_output_dir(
    monkeypatch, tmp_path, outcome, explicit_control
) -> None:
    runner = tmp_path / "gage/scripts/run/run_suite.sh"
    runner.parent.mkdir(parents=True)
    runner.touch()
    output_root = tmp_path / "results"
    user_control_root = tmp_path / "user-control"
    if explicit_control:
        monkeypatch.setenv("GAGE_SUITE_CONTROL_DIR", str(user_control_root))
        user_control_root.mkdir()
        (user_control_root / "existing.txt").write_text("preserve this file")
    else:
        monkeypatch.delenv("GAGE_SUITE_CONTROL_DIR", raising=False)
    observed_roots = []

    def fake_run(command, **kwargs):
        child_env = kwargs["env"]
        control_root = Path(child_env["GAGE_SUITE_CONTROL_DIR"])
        observed_roots.append(control_root)
        assert control_root.is_dir()
        if explicit_control:
            assert control_root == user_control_root
        else:
            assert control_root.parent == Path("/tmp")
            assert control_root.name.startswith("verl-audio-control-")
            assert "GAGE_SUITE_CONTROL_DIR" not in os.environ
        assert command[command.index("--output-dir") + 1] == str(output_root)
        run_id = command[command.index("--run-id") + 1]
        if outcome == "exception":
            raise OSError("Cannot launch suite")
        summary = output_root / run_id / "suite_summary.json"
        summary.parent.mkdir(parents=True)
        _write_summary(summary, run_id=run_id)
        return subprocess.CompletedProcess(command, 1 if outcome == "failure" else 0)

    monkeypatch.setattr(audio_benchmark.subprocess, "run", fake_run)
    config = {
        "enabled": True,
        "gage_root": str(tmp_path / "gage"),
        "data_root": str(tmp_path / "data"),
        "output_dir": str(output_root),
        "served_model_name": "model",
        "benchmarks": ["aishell2"],
    }
    kwargs = {
        "server_addresses": ["127.0.0.1:8000"],
        "model_path": "/models/test",
        "experiment_name": "audio",
        "global_step": 0,
    }
    if outcome == "success":
        assert audio_benchmark.run_audio_benchmarks(config, **kwargs)["val-audio/completed"] == 1.0
    else:
        with pytest.raises(OSError if outcome == "exception" else RuntimeError):
            audio_benchmark.run_audio_benchmarks(config, **kwargs)
    assert len(observed_roots) == 1
    if explicit_control:
        assert (user_control_root / "existing.txt").read_text() == "preserve this file"
        assert os.environ["GAGE_SUITE_CONTROL_DIR"] == str(user_control_root)
    else:
        assert not observed_roots[0].exists()
        assert "GAGE_SUITE_CONTROL_DIR" not in os.environ
    if outcome != "exception":
        assert len(list(output_root.glob("*/suite_summary.json"))) == 1


def test_audio_control_storage_closes_if_router_cannot_start(monkeypatch, tmp_path) -> None:
    runner = tmp_path / "gage/scripts/run/run_suite.sh"
    runner.parent.mkdir(parents=True)
    runner.touch()
    monkeypatch.delenv("GAGE_SUITE_CONTROL_DIR", raising=False)
    observed_roots = []
    original_storage = audio_benchmark.TemporaryDirectory

    def capture_storage(**kwargs):
        storage = original_storage(**kwargs)
        observed_roots.append(Path(storage.name))
        return storage

    class FailingRouter:
        def __init__(self, upstream_api_bases):
            pass

        def __enter__(self):
            assert observed_roots[0].is_dir()
            raise RuntimeError("Router failed to start")

        def __exit__(self, *exc):
            pass

    monkeypatch.setattr(audio_benchmark, "TemporaryDirectory", capture_storage)
    monkeypatch.setattr(audio_benchmark, "AudioBenchmarkRouter", FailingRouter)
    with pytest.raises(RuntimeError, match="Router failed to start"):
        audio_benchmark.run_audio_benchmarks(
            {
                "enabled": True,
                "gage_root": str(tmp_path / "gage"),
                "data_root": str(tmp_path / "data"),
                "output_dir": str(tmp_path / "results"),
                "served_model_name": "model",
                "benchmarks": ["aishell2"],
            },
            server_addresses=["127.0.0.1:8000", "127.0.0.1:8001"],
            model_path="/models/test",
            experiment_name="audio",
            global_step=0,
        )
    assert len(observed_roots) == 1
    assert not observed_roots[0].exists()
