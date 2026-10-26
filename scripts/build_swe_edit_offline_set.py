"""Freeze reviewed trajectories without executing any model-generated commands.

Run from verl with PYTHONPATH=.:../stock-rl-reflect using uv run --no-project.
The curated set is a regression set, not an unbiased prevalence estimate.
"""

import argparse
import gzip
import hashlib
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import yaml

from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema


def aligned_calls(text, parser, schemas):
    """Preserve dispatch order; ambiguous phases become explicit barriers."""
    ordinal = 0
    for phase in re.split(r"\nassistant\n", text):
        observations = re.findall(r"<tool_response>(.*?)</tool_response>", phase, re.S)
        source = re.sub(r"<tool_response>.*?</tool_response>", "", phase, flags=re.S)
        bodies = re.findall(r"<tool_call>\s*<function=(.*?)</function>\s*</tool_call>", source, re.S)
        if len(observations) > len(bodies) or len(observations) > 1:
            ordinal += 1
            yield dict(id=ordinal, name="__ambiguous_phase__", arguments={}, result="", historical_barrier=True)
            continue
        for body, observation in zip(bodies, observations):
            ordinal += 1
            call = parser._parse_xml_function_call(body, schemas)
            yield dict(id=ordinal, name=call.name if call else "__parse_failed__",
                       arguments=json.loads(call.arguments) if call else {}, result=observation,
                       xml="<tool_call><function=" + body + "</function></tool_call>",
                       historical_barrier=call is None)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--review", type=Path, default=Path("analysis_reports/recovery_focus_20261026"))
    ap.add_argument("--output", type=Path, default=Path("analysis_reports/edit_offline_20261026"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    annotations = json.loads((args.review / "manual_annotations.json").read_text())
    additional = args.output / "additional_annotations.json"
    if additional.exists():
        annotations.extend(json.loads(additional.read_text()))
    provenance = json.loads((args.review / "provenance.json").read_text())
    expected_hashes = {x["path"]: x["sha256"] for x in provenance["raw_inputs"]}
    groups = defaultdict(list)
    repo = Path.cwd()
    for a in annotations:
        groups[repo / a["kind"] / "three_penalty_more_cases" / f"{a['step']}.jsonl"].append(a)
    parser = Qwen3XMLToolParser(None)
    from verl.experimental.agent_loop.tool_parser import logger

    logger.setLevel(logging.ERROR)
    config_path = repo.parent / "stock-rl-reflect/recipe/swe_agent/config/tool_config_claude_like.yaml"
    config = yaml.safe_load(config_path.read_text())
    schemas = [OpenAIFunctionToolSchema.model_validate(t["tool_schema"]) for t in config["tools"]]
    cards, index = [], []
    for source, labels in sorted(groups.items()):
        wanted = {a["line"]: a for a in labels}
        hasher = hashlib.sha256()
        for line, raw in enumerate(source.open("rb"), 1):
            hasher.update(raw)
            if line not in wanted:
                continue
            label = wanted[line]
            data = json.loads(raw)
            calls = list(aligned_calls(data["output"], parser, schemas))
            edits = [c for c in calls if c["name"] == "Edit"]
            failures = [c["id"] for c in edits if any(s in c["result"] for s in (
                "old_string and new_string are identical", "old_string was not found",
                "old_string matched", "read the file first", "does not exist; use an empty old_string"))]
            card = dict(case_id=f"{label['kind']}_{label['step']}_{line}", annotation=label,
                        source=str(source), line_sha256=hashlib.sha256(raw).hexdigest(),
                        input=data["input"], output=data["output"], calls=calls,
                        observed_failed_edit_ids=failures,
                        historical_proofs_available=False, historical_generated_token_alignment_available=False)
            cards.append(card)
            index.append({k: card[k] for k in ("case_id", "source", "line_sha256", "observed_failed_edit_ids")}
                            | dict(task_id=label["task_id"], hidden_pass=label["hidden_pass"],
                                   cohort=label["cohort"], calls=len(calls), edits=len(edits)))
        assert hasher.hexdigest() == expected_hashes[str(source)], f"Source changed: {source}"
    assert len(cards) == len(annotations)
    def freeze(filename, subset):
        target = args.output / filename
        # Deterministic gzip header, so rebuilding gives the same corpus digest.
        with target.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            for card in sorted(subset, key=lambda c: c["case_id"]):
                stream.write((json.dumps(card, ensure_ascii=False) + "\n").encode())
        return hashlib.sha256(target.read_bytes()).hexdigest()

    corpus_hash = freeze("reviewed_trajectories.jsonl.gz", cards)
    repeated = [c for c in cards if len(c["observed_failed_edit_ids"]) >= 3]
    other = [c for c in cards if len(c["observed_failed_edit_ids"]) < 3]
    subsets = {
        "confirmed_repeated_edit_trajectories.jsonl.gz": dict(cases=len(repeated), sha256=freeze("confirmed_repeated_edit_trajectories.jsonl.gz", repeated)),
        "other_reviewed_controls.jsonl.gz": dict(cases=len(other), sha256=freeze("other_reviewed_controls.jsonl.gz", other)),
    }
    manifest = dict(version=1, experiment="three_penalty_more_cases", wandb_run="pxikjoll",
                    selection="36 previously reviewed cases plus explicitly reviewed exact-rejection follow-ups; Edit positives and controls",
                    source_annotations_sha256=hashlib.sha256((args.review / "manual_annotations.json").read_bytes()).hexdigest(),
                    tool_schema_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    additional_annotations_sha256=hashlib.sha256(additional.read_bytes()).hexdigest() if additional.exists() else None,
                    corpus_sha256=corpus_hash, subsets=subsets, cases=index,
                    limitations=["Purposive regression set; not population prevalence or causal task-success evidence",
                                 "Saved text contains no authoritative file snapshots or historical token alignment",
                                 "Successful trajectories may still contain positive wasted-call examples"])
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(cases=len(cards), repeated_edit_cases=sum(len(c["observed_failed_edit_ids"]) >= 3 for c in cards),
                          corpus_sha256=manifest["corpus_sha256"])))


if __name__ == "__main__":
    main()
