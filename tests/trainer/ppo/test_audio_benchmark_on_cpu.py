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
) -> None:
    child_summary = path.parent / "aishell2" / "summary.json"
    child_summary.parent.mkdir(parents=True, exist_ok=True)
    child_summary.write_text(
        json.dumps(
            {
                "metrics": [
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
                        "benchmark_id": "aishell2",
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


def test_all_supported_benchmarks_use_their_gage_flags_and_dataset_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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

    monkeypatch.setattr(audio_benchmark, "uuid4", lambda: SimpleNamespace(hex="0123456789abcdef"))

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        for flag, directory in expected.items():
            assert command[command.index(flag) + 1] == str(data_root / directory)
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
        },
        server_addresses=["127.0.0.1:8000"],
        model_path="/models/test",
        experiment_name="experiment",
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
