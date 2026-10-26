"""Freeze original trajectories for Edit and Grep productivity regression tests."""

import argparse
import gzip
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path

import yaml

from build_swe_edit_offline_set import aligned_calls
from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser, logger
from verl.tools.schemas import OpenAIFunctionToolSchema


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--output", type=Path, default=Path("analysis_reports/productivity_offline_20261026"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prior = Path("analysis_reports/edit_offline_20261026")
    audit = Path("analysis_reports/tool_productivity_20261026")
    old_manifest = json.loads((prior / "manifest.json").read_text())
    old_corpus = prior / "reviewed_trajectories.jsonl.gz"
    assert hashlib.sha256(old_corpus.read_bytes()).hexdigest() == old_manifest["corpus_sha256"]
    cards = {r["case_id"]: r for r in map(json.loads, gzip.open(old_corpus, "rt"))}
    sources = {p["path"]: p for p in json.loads((audit / "provenance.json").read_text())}
    selections = defaultdict(dict)
    schema_sessions = set()
    with gzip.open(audit / "census.jsonl.gz", "rt") as stream:
        census = [json.loads(line) for line in stream]
    for row in census:
        if row["primary"] and row["counts"].get("grep_unsupported_parameters", 0):
            selections[row["source"]][row["line"]] = dict(row, cohort="all_primary_grep_schema_errors")
            schema_sessions.add(f"{row['kind']}_{row['step']}_{row['line']}")
    for row in json.loads((audit / "reviewed_cases.json").read_text()):
        row = dict(row, kind=Path(row["source"]).parent.parent.name, cohort=row["key"])
        selections[row["source"]][row["line"]] = row
    config = Path("../stock-rl-reflect/recipe/swe_agent/config/tool_config_claude_like.yaml")
    schemas = [OpenAIFunctionToolSchema.model_validate(t["tool_schema"])
               for t in yaml.safe_load(config.read_text())["tools"]]
    parser = Qwen3XMLToolParser(None)
    logger.setLevel(logging.ERROR)
    for source, wanted in sorted(selections.items()):
        sha = hashlib.sha256()
        found = set()
        for number, raw in enumerate(Path(source).open("rb"), 1):
            sha.update(raw)
            if number not in wanted:
                continue
            row = wanted[number]
            data = json.loads(raw)
            assert data["task_id"] == row["task_id"]
            line_hash = hashlib.sha256(raw).hexdigest()
            if row.get("line_sha256"):
                assert line_hash == row["line_sha256"]
            case_id = f"{row['kind']}_{row['step']}_{number}"
            if case_id not in cards:
                calls = list(aligned_calls(data["output"], parser, schemas))
                cards[case_id] = dict(case_id=case_id, annotation={k: row.get(k) for k in (
                    "kind", "step", "line", "task_id", "hidden_pass", "cohort", "annotation")},
                    source=source, line_sha256=line_hash, input=data["input"], output=data["output"], calls=calls,
                    historical_proofs_available=False, historical_generated_token_alignment_available=False)
            else:
                assert cards[case_id]["line_sha256"] == line_hash
            found.add(number)
        assert found == set(wanted)
        assert sha.hexdigest() == sources[source]["sha256"], source
    for case_id, card in cards.items():
        card["primary_schema_session"] = case_id in schema_sessions
        calls = card["calls"]
        card["observed_identical_edit_ids"] = [c["id"] for c in calls if c["name"] == "Edit"
            and c["result"].strip().split("\n")[0] == "Edit made no changes: old_string and new_string are identical."]
        card["observed_schema_error_ids"] = [c["id"] for c in calls if c["name"] == "Grep"
            and c["result"].strip().startswith("Search error: unsupported parameters:")]
    target = args.output / "trajectories.jsonl.gz"
    with target.open("wb") as raw, gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as stream:
        for card in sorted(cards.values(), key=lambda c: c["case_id"]):
            stream.write((json.dumps(card, ensure_ascii=False) + "\n").encode())
    expected = sum(r["counts"].get("grep_unsupported_parameters", 0) for r in census if r["primary"])
    got = sum(len(c["observed_schema_error_ids"]) for c in cards.values() if c["primary_schema_session"])
    assert got == expected, (got, expected)
    manifest = dict(version=1, corpus_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        previous_corpus_sha256=old_manifest["corpus_sha256"], previous_cases=40,
        tool_schema_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        selection="Previous 40-case corpus, all primary unsupported-Grep sessions through update18, and13 reviewed productivity contexts",
        cases=len(cards), primary_schema_sessions=len(schema_sessions), primary_schema_calls=got,
        identical_edit_calls=sum(len(c["observed_identical_edit_ids"]) for c in cards.values()),
        sources=[sources[s] for s in sorted(selections)],
        index=[dict(case_id=c["case_id"], task_id=c["annotation"]["task_id"], source=c["source"],
                    line_sha256=c["line_sha256"], identical_edit_ids=c["observed_identical_edit_ids"],
                    schema_error_ids=c["observed_schema_error_ids"]) for c in sorted(cards.values(), key=lambda c: c["case_id"])])
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"sources", "index", "selection"}}))


if __name__ == "__main__":
    main()
