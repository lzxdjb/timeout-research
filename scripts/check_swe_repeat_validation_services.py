"""Read-only preflight for the separate baseline and Shell Read service pools."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-shell", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = Path(os.environ.get("SWE_SOURCE", str(root.parent / "stock-rl-reflect")))
    execution_files = ("remote_execution_service.py", "tools.py", "tool_repeat_evidence.py",
                       "tool_contract_info.py", "tool_arguments.py", "test_results.py",
                       "python_test_repeat_evidence.py")
    hashes = {name: hashlib.sha256((source / "recipe/swe_agent" / name).read_bytes()).hexdigest()
              for name in execution_files}
    token = os.environ.get("SWE_AGENT_EXECUTION_TOKEN") or os.environ.get("SWE_AGENT_REMOTE_EXECUTION_TOKEN")
    urls = list(dict.fromkeys(
        url.strip().rstrip("/") for url in os.environ["SWE_AGENT_EXECUTION_URLS"].split(",") if url.strip()
    ))
    if not urls:
        raise SystemExit("Empty execution-service pool")

    def inspect(url):
        headers = {"Authorization": "Bearer " + token} if token else {}
        opener = build_opener(ProxyHandler({}))
        try:
            with opener.open(Request(url + "/tool_contract_info", headers=headers), timeout=20) as response:
                info = json.load(response)
            private = info.get("private_diagnostics", {})
            fingerprint = info.get("source_fingerprint", {})
            files = fingerprint.get("files", {})
            errors = []
            if info.get("ok") is not True:
                errors.append("service did not return ok")
            if private.get("repeat_read_evidence_enabled") is not True:
                errors.append("dedicated-read evidence must be enabled")
            if private.get("repeat_shell_evidence_enabled") is not bool(args.expect_shell):
                errors.append("Shell Read executor flag differs from this arm")
            for name, expected in hashes.items():
                if files.get(name) != expected:
                    errors.append("source mismatch: " + name)
            return {"url": url, "service_id": private.get("service_id"),
                    "repeat_read_evidence_enabled": private.get("repeat_read_evidence_enabled"),
                    "repeat_shell_evidence_enabled": private.get("repeat_shell_evidence_enabled"),
                    "source_fingerprint": fingerprint, "errors": errors}
        except Exception as exc:
            # Do not persist response bodies, request headers or credentials.
            return {"url": url, "errors": [type(exc).__name__]}

    with ThreadPoolExecutor(max_workers=min(8, len(urls))) as pool:
        services = list(pool.map(inspect, urls))
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "experiment_name": os.environ.get("EXPERIMENT_NAME"),
              "python_test_shadow_enabled": os.environ.get("SWE_AGENT_TOOL_REPEAT_PYTHON_TEST_SHADOW_ENABLED", "0"),
              "python_test_penalty_enabled": os.environ.get("SWE_AGENT_TRAINING_TOOL_REPEAT_PENALTY_PYTHON_TESTS", "0"),
              "expected_shell_evidence": bool(args.expect_shell), "local_execution_hashes": hashes,
              "services": services, "passed": all(not item["errors"] for item in services)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Service preflight: {'passed' if result['passed'] else 'failed'}; {args.output}")
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
