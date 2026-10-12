import base64
import hashlib
import json
import re
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import swe_verified_reference_filter as verified
from scripts.swe_verified_native import native_score
from recipe.swe_agent import remote_execution_service as service
from recipe.swe_agent import tool_contract_info as info
from recipe.swe_agent.test_results import score_official_tests


def dataset(tmp_path, count=2):
    gold, hidden = tmp_path / "gold.patch", tmp_path / "hidden.patch"
    gold.write_text("diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-bad\n+good\n")
    hidden.write_text("hidden")
    rows = []
    for i in range(count):
        task_id = f"pytest-dev__pytest-{i}"
        raw = tmp_path / f"raw-{i}.json"
        f2p, p2p = ["tests/a.py::test_bug"], ["tests/a.py::test_old"]
        raw.write_text(json.dumps({"instance_id": task_id, "repo": "pytest-dev/pytest", "base_commit": "abc",
                                   "FAIL_TO_PASS": f2p, "PASS_TO_PASS": p2p}))
        task = {"task_id": task_id, "repo": "pytest-dev/pytest", "base_sha": "abc", "docker_image": "cached",
                "test_commands": ["set -eo pipefail\nsource /opt/miniconda3/bin/activate\nconda activate testbed\npytest -rA tests/a.py"],
                "hidden_test_patch_path": str(hidden), "artifact_paths": {"gold_patch": str(gold), "raw_example": str(raw)}}
        rows.append({"extra_info": {"tools_kwargs": {"__swe_task_json__": json.dumps(task)}},
                     "reward_model": {"ground_truth": json.dumps({"benchmark": "swe_bench_verified",
                         "expected": "all_official_tests_pass", "fail_to_pass": f2p, "pass_to_pass": p2p})}, "index": i})
    path = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(rows).replace_schema_metadata({b"fixture": b"unchanged"}), path)
    return path


class Client:
    def __init__(self, tmp_path, fault=None):
        self.root = tmp_path / "private"
        self.root.mkdir(exist_ok=True)
        self.patched, self.active, self.calls, self.counts = set(), set(), [], {}
        self.fault = fault

    def get(self, url, path):
        if "?" in path:
            return {"session_state": {"sessions": False}, "cleanup_state": {"complete": True}}
        return {"ok": True, "source_fingerprint": info.source_fingerprint(),
                "cached_images": [{"tag": "cached", "id": "sha256:fixture"}]}

    def post(self, url, path, payload, deadline):
        task, request = payload["task"], payload["request_id"]
        if path == "/claim":
            assert request not in self.active
            self.active.add(request)
        elif path == "/release":
            self.active.remove(request)
            return {"ok": True, "released": True}
        elif path == "/execute" and "git apply --check" in payload["parameters"]["command"]:
            self.patched.add(request)
        elif path == "/reward":
            gold = request in self.patched
            key = (task["task_id"], gold)
            n = self.counts[key] = self.counts.get(key, 0) + 1
            self.calls.append((task["task_id"], gold, request))
            if self.fault in {"transport", "transport_always"} and gold and (n == 1 or self.fault == "transport_always"):
                raise ConnectionError("fixture transient")
            session = service.ExecutionSession("key", task, request, self.root, False)
            phases = session._active_evaluator_phases = {}
            def run(command, **kwargs):
                marker = re.findall(r"SWE_RUNNER_LAUNCHED_[a-f0-9]+", command)[0]
                passed = gold and not (self.fault == "flaky" and n == 2)
                text = f"{'PASSED' if passed else 'FAILED'} tests/a.py::test_bug\nPASSED tests/a.py::test_old\n"
                text += "2 passed in 0.1s\n" if passed else "1 failed, 1 passed in 0.1s\n"
                junit = re.search(r"SWE_JUNIT_[a-f0-9]+", command)
                if junit:
                    failure = "" if passed else "<failure message='bug'/>"
                    xml = ('<testsuite tests="2" failures="' + str(int(not passed)) + '">' +
                           '<testcase classname="tests.a" name="test_bug">' + failure + '</testcase>' +
                           '<testcase classname="tests.a" name="test_old"/></testsuite>')
                    text += f"{junit[0]}_BEGIN\n{base64.b64encode(xml.encode()).decode()}\n{junit[0]}_END\n"
                return int(not passed), marker + "\n" + text
            session.run = run
            _, _, results = session._run_selected_commands({}, key="test_commands", return_results=True)
            score = score_official_tests(task["official_test_contract"],
                [r["_private_test_evidence"] for r in results], len(task["test_commands"]))
            private = score.pop("_private_official_results")
            private["score"] = dict(score)
            phases["official_tests"] = json.dumps(private)
            if self.fault == "raw_mismatch":
                phases["junit_xml_0"] = phases["junit_xml_0"].replace("test_old", "different_test")
            elif self.fault == "native_gap":
                phases["test_commands"] = phases["test_commands"].replace("PASSED tests/a.py::test_old", "unrecognized test result")
            elif self.fault == "request_mismatch":
                request = "earlier-request"
            artifact = {"task_id": task["task_id"], "request_id": request, "phases": {
                k: {"output": v, "truncated": self.fault == "truncated", "sha256": hashlib.sha256(v.encode()).hexdigest()}
                for k, v in phases.items()}}
            data = json.dumps(artifact).encode()
            sha = hashlib.sha256(data).hexdigest()
            location = self.root / (sha + ".json")
            location.write_bytes(data)
            if self.fault == "metadata":
                score["official_tests_passed"] = -1
            return {"ok": True, "score": float(score["official_tests_success"]), "metadata": {
                **score, "hidden_runner_provenance": "launch_marker_v1", "hidden_tests_started": True,
                "private_evaluator_artifact_path": str(location), "private_evaluator_artifact_sha256": sha}}
        return {"ok": True, "text": "exit_code: 0\n", "metadata": {"workspace": "/tmp/a", "container_name": "fixture"}}


def run(tmp_path, source, client, **kwargs):
    return verified.run_filter(source, ["http://fixture"], tmp_path / "audit", tmp_path / "filtered.parquet",
        roots=[f"{client.root}={client.root}"], client=client, workers=1, **kwargs)


def test_two_fresh_rounds_and_unchanged_export_and_resume(tmp_path):
    source = dataset(tmp_path)
    client = Client(tmp_path)
    report = run(tmp_path, source, client)
    assert report["strict_repeat_qualified"] == report["native_repeat_qualified"] == 2
    assert len(client.calls) == len({c[2] for c in client.calls}) == 8
    assert pq.read_table(source).equals(pq.read_table(tmp_path / "filtered.parquet"), check_metadata=True)
    assert run(tmp_path, source, client, resume=True) == report
    assert len(client.calls) == 8
    with pytest.raises(ValueError, match="overwrite"):
        run(tmp_path, source, client)
    with pytest.raises(ValueError, match="resume inputs"):
        run(tmp_path, source, client, resume=True, seconds=600)


@pytest.mark.parametrize("fault,expected,rounds", [("flaky", 0, 2), ("transport", 1, 3),
    ("transport_always", 0, 4), ("metadata", 0, 2), ("truncated", 0, 2), ("raw_mismatch", 0, 2)])
def test_failures_are_repeated_and_only_transients_retried(tmp_path, fault, expected, rounds):
    source = dataset(tmp_path, 1)
    client = Client(tmp_path, fault)
    report = run(tmp_path, source, client)
    assert report["strict_repeat_qualified"] == expected
    assert len(client.calls) == 2 * rounds
    assert pq.read_table(tmp_path / "filtered.parquet").num_rows == expected


def test_pilot_cannot_pretend_96_is_100(tmp_path):
    source = dataset(tmp_path, 96)
    client = Client(tmp_path)
    with pytest.raises(ValueError, match="100 exceeds source task count 96"):
        run(tmp_path, source, client, pilot_size=100)
    assert client.calls == []


def test_pilot_100_is_deterministic(tmp_path):
    source = dataset(tmp_path, 101)
    client = Client(tmp_path)
    report = run(tmp_path, source, client, pilot_size=100, seed=42)
    assert report["audited_tasks"] == report["strict_repeat_qualified"] == 100
    assert len(client.calls) == 400
    assert pq.read_table(tmp_path / "filtered.parquet").num_rows == 100
    again = verified.run_filter(source, ["http://fixture"], tmp_path / "other_audit", tmp_path / "other.parquet",
        roots=[f"{client.root}={client.root}"], client=client, workers=1, pilot_size=100, seed=42)
    assert again["groups"] == report["groups"]


def test_native_grading_and_missing_coverage_and_xfail():
    contract = {"fail_to_pass": ["tests/a.py::test_bug"], "pass_to_pass": ["tests/a.py::test_old"]}
    score = native_score("pytest-dev/pytest", ["XFAIL tests/a.py::test_bug\nPASSED tests/a.py::test_old"], contract)
    assert score["complete"] and score["official_resolved"]
    absent = native_score("pytest-dev/pytest", ["PASSED tests/a.py::test_old"], contract)
    assert not absent["complete"] and absent["fail_to_pass_failed"] == 0
    assert absent["missing"] == ["tests/a.py::test_bug"]


def good_attempt(prefix, strict=True, native=True):
    arms = {}
    for mode in ("baseline", "gold"):
        arms[mode] = {"request_id": prefix + mode, "cleanup_verified": True}
        for parser, valid in [("strict", strict), ("native", native)]:
            arms[mode][parser] = {"valid": True, "complete": valid, "fail_to_pass_failed": int(mode == "baseline"),
                                  "pass_to_pass_complete": True, "all_required_passed": mode == "gold"}
    return {"image_id": "image", "arms": arms}


def test_parser_qualification_independent_and_transient_does_not_hide_failure():
    a, b = good_attempt("a", strict=False), good_attempt("b", strict=False)
    proof = verified.qualification([a, b], 2)
    assert not proof["strict"]["qualified"] and proof["native"]["qualified"]
    b["arms"]["gold"] = {"transient": True, "request_id": "bgold", "cleanup_verified": True}
    assert verified.attempt_state(b, "strict") == "veto"
    assert verified.attempt_state(b, "native") == "transient"


def test_request_reuse_and_image_changes_rejected():
    a, b = good_attempt("a"), good_attempt("b")
    with pytest.raises(ValueError, match="reuses"):
        verified.qualification([a, a], 2)
    b["image_id"] = "different"
    with pytest.raises(ValueError, match="image identity"):
        verified.qualification([a, b], 2)


def test_export_rejects_changed_receipt(tmp_path):
    source = dataset(tmp_path, 1)
    client = Client(tmp_path)
    run(tmp_path, source, client)
    (tmp_path / "filtered.parquet.audit.json").write_text("{}")
    with pytest.raises(ValueError, match="overwrite"):
        run(tmp_path, source, client, resume=True)


def test_native_missing_coverage_does_not_remove_strict_data(tmp_path):
    source = dataset(tmp_path, 1)
    client = Client(tmp_path, "native_gap")
    report = run(tmp_path, source, client)
    assert report["strict_repeat_qualified"] == 1
    assert report["native_repeat_qualified"] == 0
    assert report["counts"]["strict_only"] == 1
    assert report["rounds"][0]["native_gold_official_resolved"] == 0


def test_wrong_request_private_evidence_cannot_certify(tmp_path):
    source = dataset(tmp_path, 1)
    client = Client(tmp_path, "request_mismatch")
    report = run(tmp_path, source, client)
    assert report["strict_repeat_qualified"] == report["native_repeat_qualified"] == 0
    records = [json.loads(s) for s in (tmp_path / "audit/comparison_results.jsonl").read_text().splitlines()]
    assert records[0]["attempts"][0]["arms"]["gold"]["reason"] == "private_evidence_request_mismatch"


def test_resume_after_first_round_does_not_reexecute_it(tmp_path, monkeypatch):
    source = dataset(tmp_path, 1)
    client = Client(tmp_path)
    original = verified.audit.reference_audit
    calls = 0
    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted between rounds")
        return original(*args, **kwargs)
    monkeypatch.setattr(verified.audit, "reference_audit", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        run(tmp_path, source, client)
    assert len(client.calls) == 2
    assert not (tmp_path / "filtered.parquet").exists()
    monkeypatch.setattr(verified.audit, "reference_audit", original)
    assert run(tmp_path, source, client, resume=True)["strict_repeat_qualified"] == 1
    assert len(client.calls) == 4


def test_vendor_checksum_catches_source_change(tmp_path, monkeypatch):
    from scripts import swe_verified_native as native
    location = tmp_path / "vendor"
    shutil.copytree(native.VENDOR, location)
    monkeypatch.setattr(native, "VENDOR", location)
    with (location / "python.py").open("a") as stream:
        stream.write("\n# mutation\n")
    with pytest.raises(ValueError, match="source checksum"):
        native.verify_vendor()
