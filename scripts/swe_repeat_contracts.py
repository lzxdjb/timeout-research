"""Small, explicitly labelled boundary cases derived from the trajectory audit.

These are policy specifications, not expected values computed by a detector.
They are frozen into each reference-set version alongside the real trajectories.
"""

from __future__ import annotations

import copy
import hashlib
import tempfile
from pathlib import Path


def event(name, arguments, response="ok", **extra):
    return {"name": name, "arguments": arguments, "response": response, **extra}


def contracts():
    read = event("Read", {"file_path": "src/a.py"}, "VALUE = 1")
    edit = event("Edit", {"file_path": "src/a.py", "old_string": "1", "new_string": "2"}, "Edited src/a.py (1 replacement).")
    failed = event("Edit", {"file_path": "src/a.py", "old_string": "absent", "new_string": "2"}, "Edit failed: old_string was not found in src/a.py.")
    noop = event("Edit", {"file_path": "src/a.py", "old_string": "1", "new_string": "1"}, "Edit made no changes: old_string and new_string are identical.")
    test = event("Bash", {"command": "python -m pytest tests/test_a.py -v"}, "exit_code: 1\n1 failed")
    flagged = event("Bash", {"command": "python -m pytest tests/test_a.py -v", "verification": True}, "exit_code: 1\n1 failed")
    transport = event("Read", {"file_path": "src/a.py"}, "Remote execution error: service unavailable")
    search = event("Grep", {"pattern": "symbol", "path": "src"}, "No matches found.")
    missing = event("Grep", {"pattern": "symbol", "path": "src"}, "Search unavailable: rg is not installed.")
    specs = [
        ("same_file_read_after_actual_edit", [read, edit, read], [False, False, False], [0, 1, 0]),
        ("failed_edit_cannot_exempt_read", [read, failed, read], [False, False, True], [0, 0, 0]),
        ("noop_edit_cannot_exempt_read", [read, noop, read], [False, False, True], [0, 0, 0]),
        ("unrelated_edit_cannot_exempt_read", [read, event("Edit", {"file_path": "src/b.py", "old_string": "1", "new_string": "2"}, "Edited src/b.py (1 replacement)."), read], [False, False, True], [0, 1, 0]),
        ("ordinary_pytest_after_edit", [test, edit, test], [False, False, False], [0, 1, 0]),
        ("explicit_verification_after_edit", [flagged, edit, flagged], [False, False, False], [0, 1, 0]),
        ("failed_edit_cannot_exempt_test", [flagged, failed, flagged], [False, False, True], [0, 0, 0]),
        ("noop_edit_cannot_exempt_test", [flagged, noop, flagged], [False, False, True], [0, 0, 0]),
        ("failed_test_is_not_transport_retry", [test, test], [False, True], [0, 0]),
        ("one_transport_retry", [transport, read], [False, False], [0, 0]),
        ("second_failed_transport_retry", [transport, transport, transport], [False, False, True], [0, 0, 0]),
        ("unchanged_search", [search, search], [False, True], [0, 0]),
        ("permanent_missing_capability_loop", [missing, missing, missing], [False, True, True], [0, 0, 0]),
        ("inflight_duplicate", [{**read, "batch": 0}, {**read, "batch": 0}], [False, False], [0, 0]),
        ("inflight_then_completed_duplicate", [{**read, "batch": 0}, {**read, "batch": 0}, {**read, "batch": 1}], [False, False, True], [0, 0, 0]),
        ("shell_edge_whitespace_only", [event("Bash", {"command": "  cat src/a.py\n"}), event("Bash", {"command": "cat src/a.py"})], [False, True], [0, 0]),
        ("quoted_whitespace_is_significant", [event("Bash", {"command": "printf 'a  b'"}), event("Bash", {"command": "printf 'a b'"})], [False, False], [0, 0]),
        ("string_false_is_not_verification", [event("Bash", {"command": "printf hello", "verification": "false"}), edit, event("Bash", {"command": "printf hello", "verification": "false"})], [False, False, True], [0, 1, 0]),
    ]
    for gap in [8, 16, 64]:
        fillers = [event("Read", {"file_path": f"src/filler_{i}.py"}) for i in range(gap)]
        specs.append((f"distance_{gap}_does_not_erase_history", [read, *fillers, read], [False] * (gap + 1) + [True], [0] * (gap + 2)))
    results = []
    for name, events, hits, progress in specs:
        # A contract may intentionally reuse the same event object (for
        # example, read -> edit -> read). Copy each occurrence independently;
        # deepcopying the list as a whole preserves aliases and would make the
        # two reads share whichever index was assigned last.
        events = [
            {
                **copy.deepcopy(event),
                "index": index,
                "batch": event.get("batch", index),
            }
            for index, event in enumerate(events)
        ]
        results.append({"id": name, "events": events, "expected_hits": hits, "expected_progress": progress, "kind": "explicit_policy_boundary", "derived_from": "20261007 trajectory audit; handcrafted counterfactual, not an observed rollout"})
    return results


def filesystem_evidence():
    """CPU-only ground truth for changed/no-op/failed writes, no historical commands."""
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    with tempfile.TemporaryDirectory(prefix="swe-repeat-evidence-") as tmp:
        path = Path(tmp) / "a.py"
        path.write_text("VALUE = 1\n")
        before = sha(path)
        path.write_text("VALUE = 2\n")
        changed = sha(path)
        path.write_text("VALUE = 2\n")
        noop = sha(path)
        content = path.read_text()
        missing_old_string = "VALUE = 99"
        if missing_old_string in content:
            raise AssertionError("Invalid failed-edit fixture")
        failed = sha(path)
    return {"before": before, "changed": changed, "noop": noop, "failed": failed,
            "passed": before != changed and changed == noop == failed,
            "scope": "Local content-hash contract only; does not validate remote executor instrumentation"}
