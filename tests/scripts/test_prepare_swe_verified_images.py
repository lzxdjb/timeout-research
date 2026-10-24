import json

from scripts import prepare_swe_verified_images as prepare


def manifest(tmp_path):
    value = {"identity": {
        "manifest": {"cases": [{"docker_image": image} for image in ("cached", "a", "b", "a", "c")]},
        "services": [
            {"url": "http://one", "images": {"cached": "sha256:cached"}},
            {"url": "http://two", "image_availability": {"b": False}},
        ],
    }}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    return path


class Client:
    def __init__(self, *, full=False, failure=False):
        self.calls = []
        self.full = full
        self.failure = failure

    def get(self, url, path):
        return {"ok": True, "docker_available": True, "workspace_admission": {"ok": not self.full}}

    def post(self, url, path, payload, deadline):
        self.calls.append((url, path, payload))
        if self.failure and payload["image"] == "a":
            return {"ok": False, "failure_kind": "registry_unavailable"}
        return {"ok": True, "id": "sha256:prepared"}


def test_plan_only_has_no_side_effects(tmp_path):
    client = Client()
    output = tmp_path / "receipts"
    report = prepare.prepare(manifest(tmp_path), output, client=client)
    assert report["missing_images"] == 3
    assert report["per_endpoint"] == {"http://one": 2, "http://two": 1}
    assert not output.exists()
    assert not client.calls


def test_preparation_pulls_each_missing_image_once_and_records_failures(tmp_path):
    client = Client(failure=True)
    output = tmp_path / "receipts"
    report = prepare.prepare(manifest(tmp_path), output, client=client, execute=True)
    assert sorted(call[2]["image"] for call in client.calls) == ["a", "b", "c"]
    assert all(call[1] == "/audit_prepare_image" for call in client.calls)
    assert report["prepared"] == 2
    assert report["failed"] == 1
    assert report["unattempted"] == 0
    assert not report["complete"]
    assert json.loads((output / "report.json").read_text()) == report
    receipts = [json.loads(path.read_text()) for path in output.glob("*.json") if path.name != "report.json"]
    assert len(receipts) == 3
    assert any(r.get("response", {}).get("failure_kind") == "registry_unavailable" for r in receipts)


def test_full_endpoints_stop_without_sending_pulls(tmp_path):
    client = Client(full=True)
    report = prepare.prepare(manifest(tmp_path), tmp_path / "receipts", client=client, execute=True)
    assert not client.calls
    assert report["failed"] == 2
    assert report["unattempted"] == 1
    assert not report["complete"]


def test_positive_availability_is_enough_to_skip_an_image(tmp_path):
    path = manifest(tmp_path)
    value = json.loads(path.read_text())
    value["identity"]["services"][1]["image_availability"]["a"] = True
    assert prepare.missing_images(value) == ["b", "c"]
