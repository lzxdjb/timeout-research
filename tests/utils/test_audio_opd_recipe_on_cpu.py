# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Check recipe orchestration without downloads, media inference, or GPU imports."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE = REPO_ROOT / "scripts/audio_data/prepare_opd.sh"

# Replace only module commands. The recipe's small inline Python helpers still
# run normally so atomic mixing and the empty-hypothesis guard are exercised.
FAKE_PYTHON = r"""
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
with Path(os.environ["AUDIO_TEST_LOG"]).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(args) + "\n")
if args and args[0] == "-":
    os.execv(sys.executable, [sys.executable, *args])
if len(args) < 2 or args[0] != "-m":
    raise SystemExit("unexpected recipe invocation")
module = args[1].rsplit(".", 1)[-1]
def option(name):
    return args[args.index(name) + 1]
def write_output(text):
    output = Path(option("--output"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
if module == "download":
    pass
elif module == "normalize":
    dataset = option("--dataset")
    row = {
        "sample_id": dataset + ":fixture",
        "dataset": dataset,
        "split": "train",
        "task": "asr" if dataset in {"librispeech", "aishell1"} else "closed_qa",
        "qc": {"offline": {"decision": "accept"}},
    }
    write_output(json.dumps(row) + "\n")
elif module == "filter_offline":
    write_output(Path(option("--input")).read_text(encoding="utf-8"))
elif module == "transcribe_qwen":
    text = ""
    if os.environ.get("AUDIO_TEST_EMPTY_ASR") != "1":
        text = json.dumps({"sample_id": "fixture", "text": "fixture transcript"}) + "\n"
    write_output(text)
elif module == "convert_verl":
    write_output("fixture parquet output\n")
else:
    raise SystemExit("unexpected recipe module: " + module)
"""


@pytest.fixture
def recipe_environment(tmp_path):
    python = tmp_path / "fake-python"
    python.write_text(f"#!{sys.executable}\n" + FAKE_PYTHON, encoding="utf-8")
    python.chmod(0o755)
    environment = {name: value for name, value in os.environ.items() if not name.startswith("AUDIO_")}
    environment.update(
        AUDIO_PYTHON=str(python),
        AUDIO_ROOT=str(tmp_path / "audio root with spaces"),
        AUDIO_TEST_LOG=str(tmp_path / "calls.jsonl"),
    )
    return environment


def _run_recipe(environment, stage, **overrides):
    environment = dict(environment, **overrides)
    result = subprocess.run(
        ["bash", str(RECIPE), stage],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    path = Path(environment["AUDIO_TEST_LOG"])
    calls = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return result, [args for args in calls if args[:1] == ["-m"]]


def _option(args, name):
    return args[args.index(name) + 1]


def test_default_plan_does_not_request_execution(recipe_environment):
    result, calls = _run_recipe(recipe_environment, "plan")

    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    assert calls[0][1] == "scripts.audio_data.download"
    assert "--execute" not in calls[0]
    assert "--extract" not in calls[0]
    assert _option(calls[0], "--connections") == "8"
    assert _option(calls[0], "--mirror") == "original"
    assert _option(calls[0], "--root") == str(Path(recipe_environment["AUDIO_ROOT"]) / "raw")
    assert not Path(recipe_environment["AUDIO_ROOT"]).exists()


def test_all_stages_preserve_policy_and_use_asr_checked_export(recipe_environment):
    root = Path(recipe_environment["AUDIO_ROOT"])
    holdout = root / "benchmark recordings.jsonl"
    result, calls = _run_recipe(
        recipe_environment,
        "all",
        AUDIO_HOLDOUT_MANIFEST=str(holdout),
        AUDIO_MAX_DURATION="17.5",
    )

    assert result.returncode == 0, result.stderr
    stages = [
        line.removeprefix("Audio preparation stage: ")
        for line in result.stderr.splitlines()
        if line.startswith("Audio preparation stage: ")
    ]
    assert stages == ["download", "normalize", "offline", "asr", "asr-filter", "convert"]
    assert [call[1].rsplit(".", 1)[-1] for call in calls] == [
        "download",
        "normalize",
        "normalize",
        "normalize",
        "filter_offline",
        "transcribe_qwen",
        "filter_offline",
        "convert_verl",
    ]
    mixed = [json.loads(line) for line in (root / "pilot.normalized.jsonl").read_text().splitlines()]
    assert {row["dataset"] for row in mixed} == {"librispeech", "aishell1", "clotho_aqa"}
    clotho = next(
        call
        for call in calls
        if "--dataset" in call and _option(call, "--dataset") == "clotho_aqa" and call[1].endswith(".normalize")
    )
    assert _option(clotho, "--max-samples") == "0"

    offline_calls = [call for call in calls if call[1].endswith(".filter_offline")]
    for call in offline_calls:
        assert "--allow-noncommercial" in call
        assert _option(call, "--max-duration") == "17.5"
        assert _option(call, "--holdout-manifest") == str(holdout)
    assert "--asr-hypotheses" not in offline_calls[0]
    assert _option(offline_calls[1], "--asr-hypotheses") == str(root / "qwen_asr.jsonl")
    assert _option(offline_calls[1], "--reference-check-split") == "train"
    assert "--require-reference-check" in offline_calls[1]
    for call in calls:
        if call[1].endswith((".download", ".normalize", ".filter_offline")):
            assert _option(call, "--progress") == "on"
    assert _option(offline_calls[1], "--checkpoint-dir") == str(root / "checkpoints/asr-filter")
    assert _option(offline_calls[1], "--checkpoint-interval") == "1000"
    assert "--reset-checkpoint" not in offline_calls[1]
    asr = next(call for call in calls if call[1].endswith(".transcribe_qwen"))
    assert _option(asr, "--split") == "train"
    assert _option(asr, "--max-duration") == "17.5"
    assert _option(asr, "--concurrency") == "2"
    assert _option(asr, "--progress") == "on"
    assert _option(asr, "--progress-interval") == "2"
    assert _option(calls[-1], "--input") == str(root / "pilot.asr_checked.jsonl")
    assert _option(calls[-1], "--output") == str(root / "train.opd.parquet")
    assert _option(calls[-1], "--mode") == "opd"
    assert _option(calls[-1], "--split") == "train"


def test_no_asr_hypotheses_stops_all_before_filter_and_export(recipe_environment):
    result, calls = _run_recipe(recipe_environment, "all", AUDIO_TEST_EMPTY_ASR="1")

    assert result.returncode != 0
    assert "No ASR hypotheses were produced" in result.stderr
    assert calls[-1][1] == "scripts.audio_data.transcribe_qwen"
    assert sum(call[1].endswith(".filter_offline") for call in calls) == 1
    assert not any(call[1].endswith(".convert_verl") for call in calls)


def test_asr_progress_and_concurrency_overrides_reach_the_producer(recipe_environment):
    root = Path(recipe_environment["AUDIO_ROOT"])
    root.mkdir()
    (root / "pilot.offline.jsonl").write_text("{}\n", encoding="utf-8")
    result, calls = _run_recipe(
        recipe_environment,
        "asr",
        AUDIO_QWEN_CONCURRENCY="8",
        AUDIO_PROGRESS="off",
        AUDIO_ASR_PROGRESS_INTERVAL="5",
    )
    assert result.returncode == 0, result.stderr
    asr = calls[0]
    assert _option(asr, "--concurrency") == "8"
    assert _option(asr, "--progress") == "off"
    assert _option(asr, "--progress-interval") == "5"


def test_all_five_sources_receive_explicit_download_and_extraction_caps(recipe_environment):
    datasets = "librispeech aishell1 clotho_aqa fsd50k nsynth"
    result, calls = _run_recipe(
        recipe_environment,
        "download",
        AUDIO_DATASETS=datasets,
        AUDIO_MAX_DOWNLOAD_BYTES="80000000000",
        AUDIO_MAX_EXTRACTED_BYTES="160000000000",
    )

    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    selected = [calls[0][index + 1] for index, value in enumerate(calls[0]) if value == "--dataset"]
    assert selected == datasets.split()
    assert "--execute" in calls[0]
    assert "--extract" in calls[0]
    assert _option(calls[0], "--max-download-bytes") == "80000000000"
    assert _option(calls[0], "--max-extracted-bytes") == "160000000000"
    assert _option(calls[0], "--connections") == "8"
    assert _option(calls[0], "--mirror") == "original"


@pytest.mark.parametrize("stage", ["plan", "download"])
def test_download_connection_and_official_mirror_overrides_are_forwarded(recipe_environment, stage):
    result, calls = _run_recipe(
        recipe_environment,
        stage,
        AUDIO_DOWNLOAD_CONNECTIONS="4",
        AUDIO_DOWNLOAD_MIRROR="elda",
    )
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    assert _option(calls[0], "--connections") == "4"
    assert _option(calls[0], "--mirror") == "elda"


@pytest.mark.parametrize(
    "variable,value,flag",
    [
        ("AUDIO_DOWNLOAD_CONNECTIONS", "0", "connections"),
        ("AUDIO_DOWNLOAD_CONNECTIONS", "invalid", "connections"),
        ("AUDIO_DOWNLOAD_MIRROR", "unsupported", "mirror"),
    ],
)
def test_invalid_download_settings_are_rejected_by_real_cli_before_network(recipe_environment, variable, value, flag):
    result, calls = _run_recipe(
        recipe_environment,
        "plan",
        AUDIO_PYTHON=sys.executable,
        **{variable: value},
    )
    assert result.returncode != 0
    assert flag in result.stderr
    assert "unrecognized arguments" not in result.stderr
    assert not calls
    assert not Path(recipe_environment["AUDIO_ROOT"]).exists()


@pytest.mark.parametrize("mirror,host", [("original", "www.openslr.org"), ("elda", "openslr.elda.org")])
def test_real_wrapper_plan_selects_only_the_official_openslr_mirror(recipe_environment, mirror, host):
    result, calls = _run_recipe(
        recipe_environment,
        "plan",
        AUDIO_PYTHON=sys.executable,
        AUDIO_DOWNLOAD_MIRROR=mirror,
    )
    assert result.returncode == 0, result.stderr
    assert not calls
    plan = json.loads(result.stdout)["plan"]
    for entry in plan:
        urls = [asset["url"] for asset in entry["assets"]]
        if entry["dataset"] in {"librispeech", "aishell1"}:
            assert all(url.startswith(f"https://{host}/resources/") for url in urls)
        else:
            assert all(url.startswith("https://zenodo.org/") for url in urls)
    assert not Path(recipe_environment["AUDIO_ROOT"]).exists()


def test_audio_only_conversion_has_a_separate_explicit_output(recipe_environment):
    result, calls = _run_recipe(recipe_environment, "convert-offline")
    root = Path(recipe_environment["AUDIO_ROOT"])

    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    assert _option(calls[0], "--input") == str(root / "pilot.offline.jsonl")
    assert _option(calls[0], "--output") == str(root / "train.audio_only.opd.parquet")


def test_concurrent_runner_for_same_root_fails_before_work(recipe_environment):
    fcntl = pytest.importorskip("fcntl")
    root = Path(recipe_environment["AUDIO_ROOT"])
    root.mkdir()
    with (root / ".prepare_opd.lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result, calls = _run_recipe(recipe_environment, "download")

    assert result.returncode != 0
    assert "Another preparation stage is running" in result.stderr
    assert not calls


def test_asr_filter_checkpoint_controls_are_passed_with_spaces_preserved(recipe_environment):
    root = Path(recipe_environment["AUDIO_ROOT"])
    root.mkdir()
    (root / "pilot.offline.jsonl").write_text("{}\n", encoding="utf-8")
    checkpoint = str(root / "custom checkpoint")
    result, calls = _run_recipe(
        recipe_environment,
        "asr-filter",
        AUDIO_PROGRESS="off",
        AUDIO_ASR_FILTER_CHECKPOINT_DIR=checkpoint,
        AUDIO_CHECKPOINT_EVERY="17",
        AUDIO_RESET_ASR_FILTER_CHECKPOINT="1",
    )
    assert result.returncode == 0, result.stderr
    assert _option(calls[0], "--progress") == "off"
    assert _option(calls[0], "--checkpoint-dir") == checkpoint
    assert _option(calls[0], "--checkpoint-interval") == "17"
    assert "--reset-checkpoint" in calls[0]


@pytest.mark.parametrize("variable", ["AUDIO_PROGRESS", "AUDIO_RESET_ASR_FILTER_CHECKPOINT"])
def test_invalid_progress_or_reset_modes_fail_before_work(recipe_environment, variable):
    result, calls = _run_recipe(recipe_environment, "asr-filter", **{variable: "invalid"})
    assert result.returncode != 0
    assert variable in result.stderr
    assert not calls
