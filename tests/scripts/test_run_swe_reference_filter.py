"""Check the shell entry point preserves the existing concurrency settings."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("pilot", [False, True])
@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("upgrade", [False, True])
def test_filter_uses_per_task_pipeline_and_preserves_settings(tmp_path, pilot, resume, upgrade):
    repo = Path(__file__).resolve().parents[2]
    binary = tmp_path / "bin"
    binary.mkdir()
    shim = binary / "python3"
    shim.write_text(f"#!{sys.executable}\n" + "import json, os, sys\n" +
                    "open(os.environ['FILTER_TEST_CALLS'], 'w').write(json.dumps(sys.argv[1:]))\n")
    shim.chmod(0o755)
    calls = tmp_path / "calls.json"
    env = {**os.environ, "PATH": f"{binary}:{os.environ['PATH']}",
           "SWE_AGENT_EXECUTION_URLS": "http://test-unused", "SWE_REFERENCE_EVALUATOR_ROOT": "/service=/local",
           "FILTER_TEST_CALLS": str(calls)}
    source, root, output = [tmp_path / s for s in ["source.parquet", "audit", "filtered.parquet"]]
    command = ["bash", "scripts/run_swe_reference_filter.sh", str(source), str(root), str(output)]
    if pilot:
        command += ["--pilot-size", "100", "--seed", "20261010"]
    if resume:
        command += ["--resume"]
    if upgrade:
        command += ["--resume-cleanup-upgrade"]
    result = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = json.loads(calls.read_text())
    assert args[0] == "scripts/swe_rebench_reference_filter.py"
    def value(flag):
        return args[args.index(flag) + 1]
    assert value("--dataset") == str(source)
    assert value("--output-dir") == str(root)
    assert value("--filtered-output") == str(output)
    assert value("--workers-per-endpoint") == "32"
    assert value("--image-prepare-parallelism") == "8"
    assert value("--image-prepare-deadline") == "1800"
    assert value("--scenario-deadline") == "900"
    assert value("--image-cleanup") == "task-images"
    assert value("--pilot-size") == ("100" if pilot else "0")
    assert ("--resume" in args) == resume
    assert ("--resume-cleanup-upgrade" in args) == upgrade
