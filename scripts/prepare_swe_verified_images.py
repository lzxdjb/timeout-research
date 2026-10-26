"""Prepare images absent from every endpoint in a SWE-verified audit manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import swe_tool_online_benchmark as audit


def missing_images(manifest):
    identity = manifest["identity"]
    services = identity["services"]
    images = sorted({case["docker_image"] for case in identity["manifest"]["cases"]})
    # True is positive cache evidence; unknown is not proof of availability.
    return [image for image in images if not any(
        service.get("image_availability", {}).get(image) is True
        or image in service.get("images", {}) for service in services)]


def prepare(manifest_path, output, *, execute=False, seconds=1800, client=None):
    manifest = json.loads(manifest_path.read_text())
    urls = [service["url"] for service in manifest["identity"]["services"]]
    images = missing_images(manifest)
    if not urls or len(set(urls)) != len(urls):
        raise ValueError("audit manifest must contain unique endpoints")
    jobs = {url: images[index::len(urls)] for index, url in enumerate(urls)}
    plan = {"manifest": str(manifest_path.resolve()), "missing_images": len(images),
            "per_endpoint": {url: len(tasks) for url, tasks in jobs.items()}, "execute": execute}
    print(json.dumps(plan), flush=True)
    if not execute:
        return plan
    if output.exists():
        raise ValueError("output directory exists; choose a new preparation directory")
    output.mkdir(parents=True)
    client = client or audit.Client()

    def worker(url):
        results = []
        for image in jobs[url]:
            try:
                health = client.get(url, "/status")
                if (health.get("ok") is not True or health.get("docker_available") is not True
                        or health.get("workspace_admission", {}).get("ok") is not True):
                    raise RuntimeError("endpoint unavailable or storage reserve reached: " + str(health)[:600])
                value = client.post(url, "/audit_prepare_image", {"image": image, "timeout_seconds": seconds},
                                    time.monotonic() + seconds + 10)
                ok = value.get("ok") is True and bool(value.get("id"))
                result = {"endpoint": url, "image": image, "ok": ok, "response": value}
            except Exception as exc:
                result = {"endpoint": url, "image": image, "ok": False,
                          "error": f"{type(exc).__name__}: {exc}"[:1500]}
            audit.audit_write_json(output / (audit.digest([url, image]) + ".json"), result)
            print(json.dumps(result), flush=True)
            results.append(result)
            if not result["ok"] and "storage reserve" in result.get("error", ""):
                # Avoid repeatedly sending pulls to a full/unhealthy daemon.
                break
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(urls)) as pool:
        results = [r for group in pool.map(worker, urls) for r in group]
    report = {**plan, "prepared": sum(r["ok"] for r in results),
              "failed": sum(not r["ok"] for r in results), "unattempted": len(images) - len(results),
              "complete": len(results) == len(images) and all(r["ok"] for r in results)}
    audit.audit_write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execute", action="store_true", help="actually prepare images; default only prints plan")
    parser.add_argument("--deadline", type=float, default=1800)
    args = parser.parse_args()
    if not 0 < args.deadline <= 7200:
        parser.error("--deadline must be in (0, 7200]")
    report = prepare(args.audit_manifest, args.output_dir, execute=args.execute, seconds=args.deadline)
    print(json.dumps(report, indent=2))
    if args.execute and not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
