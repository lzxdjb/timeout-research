from __future__ import annotations

import io
import threading
import urllib.error

from recipe.swe_agent.execution_router import ExecutionServerRouter, _ServerState
from verl.trainer.ppo import swe_image_prefetch as ray_trainer
from verl.trainer.ppo.swe_image_prefetch import _SWETrainingImagePrefetcher, _SWEValidationImagePrefetcher
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _tools_kwargs(image: str):
    return {"__swe_task_json__": f'{{"task_id":"{image}","docker_image":"{image}"}}'}


class _Frame:
    column_names = ("extra_info",)

    def __init__(self, images):
        self.extra_infos = [{"tools_kwargs": _tools_kwargs(image)} for image in images]

    def __getitem__(self, key):
        assert key == "extra_info"
        return self.extra_infos


class _Loader:
    def __init__(self, images):
        self.dataset = type("Dataset", (), {"dataframe": _Frame(images)})()


def _enable(monkeypatch, *, chunk_size=16, parallelism=2):
    monkeypatch.setenv("SWE_AGENT_VALIDATION_IMAGE_PREFETCH", "1")
    monkeypatch.setenv("SWE_AGENT_EXECUTION_URL", "http://execution.test:18080")
    monkeypatch.delenv("SWE_AGENT_EXECUTION_URLS", raising=False)
    monkeypatch.setenv("SWE_AGENT_VALIDATION_IMAGE_PREFETCH_CHUNK_SIZE", str(chunk_size))
    monkeypatch.setenv("SWE_AGENT_VALIDATION_IMAGE_PREFETCH_PARALLELISM", str(parallelism))


def _enable_training(monkeypatch, *, chunk_size=16, parallelism=2):
    monkeypatch.setenv("SWE_AGENT_TRAINING_IMAGE_PREFETCH", "1")
    monkeypatch.setenv("SWE_AGENT_EXECUTION_URL", "http://execution.test:18080")
    monkeypatch.delenv("SWE_AGENT_EXECUTION_URLS", raising=False)
    monkeypatch.setenv("SWE_AGENT_TRAINING_IMAGE_PREFETCH_CHUNK_SIZE", str(chunk_size))
    monkeypatch.setenv("SWE_AGENT_TRAINING_IMAGE_PREFETCH_PARALLELISM", str(parallelism))


class _StatefulLoader:
    def __init__(self, values):
        self.values = values
        self.position = 0

    def __iter__(self):
        while self.position < len(self.values):
            value = self.values[self.position]
            self.position += 1
            yield value

    def state_dict(self):
        return {"position": self.position}


def test_training_lookahead_preserves_state_after_current_batch():
    loader = _StatefulLoader(["current", "next", "last"])

    batches = list(RayPPOTrainer._iter_batches_with_lookahead(loader, enabled=True, capture_state=True))

    assert batches == [
        ("current", "next", {"position": 1}),
        ("next", "last", {"position": 2}),
        ("last", None, {"position": 3}),
    ]


def test_disabled_lookahead_does_not_consume_the_next_batch_early():
    loader = _StatefulLoader(["current", "next"])
    iterator = RayPPOTrainer._iter_batches_with_lookahead(loader, enabled=False, capture_state=True)

    assert next(iterator) == ("current", None, None)
    assert loader.position == 1


def test_training_prefetch_configuration_is_independent_from_validation(monkeypatch):
    _enable_training(monkeypatch, chunk_size=24, parallelism=3)
    monkeypatch.delenv("SWE_AGENT_VALIDATION_IMAGE_PREFETCH", raising=False)

    training_prefetcher = _SWETrainingImagePrefetcher()
    validation_prefetcher = _SWEValidationImagePrefetcher()

    assert training_prefetcher.enabled is True
    assert training_prefetcher.chunk_size == 24
    assert training_prefetcher.parallelism == 3
    assert validation_prefetcher.enabled is False
    training_prefetcher.close()
    validation_prefetcher.close()


def test_training_batch_submission_is_nonblocking_and_deduplicates_images(monkeypatch):
    _enable_training(monkeypatch)
    prefetcher = _SWETrainingImagePrefetcher()
    started = threading.Event()
    release = threading.Event()

    def blocked_prefetch(images, _batch_index, _label):
        started.set()
        assert release.wait(timeout=5)
        return {"ok": True, "images": images}

    monkeypatch.setattr(prefetcher, "_prefetch", blocked_prefetch)
    batch = {
        "tools_kwargs": [
            _tools_kwargs("image:a"),
            _tools_kwargs("image:a"),
            _tools_kwargs("image:b"),
        ]
    }

    assert prefetcher.submit_batch(batch, batch_index=1, label="step-1") == 2
    assert started.wait(timeout=2)
    with prefetcher._lock:
        assert prefetcher._pending_images == {"image:a", "image:b"}

    release.set()
    prefetcher.wait(timeout=5)
    assert prefetcher._accepted_images == {"image:a", "image:b"}
    prefetcher.close()


def test_async_training_submission_is_bounded_and_uses_lazy_claim_fallback(monkeypatch):
    _enable_training(monkeypatch, chunk_size=1)
    prefetcher = _SWETrainingImagePrefetcher()
    started = threading.Event()
    release = threading.Event()

    def blocked_prefetch(images, _batch_index, _label):
        started.set()
        assert release.wait(timeout=5)
        return {"ok": True, "images": images}

    monkeypatch.setattr(prefetcher, "_prefetch", blocked_prefetch)
    batch = {
        "tools_kwargs": [
            _tools_kwargs("image:a"),
            _tools_kwargs("image:b"),
            _tools_kwargs("image:c"),
        ]
    }

    assert (
        prefetcher.submit_batch_bounded(
            batch,
            batch_index=1,
            label="sample-1",
            max_pending_images=2,
        )
        == 2
    )
    assert started.wait(timeout=2)
    with prefetcher._lock:
        assert prefetcher._pending_images == {"image:a", "image:b"}

    assert (
        prefetcher.submit_batch_bounded(
            {"tools_kwargs": [_tools_kwargs("image:c")]},
            batch_index=2,
            label="sample-2",
            max_pending_images=2,
        )
        == 0
    )

    release.set()
    prefetcher.wait(timeout=5)
    assert prefetcher._accepted_images == {"image:a", "image:b"}
    assert (
        prefetcher.submit_batch_bounded(
            {"tools_kwargs": [_tools_kwargs("image:c")]},
            batch_index=3,
            label="sample-3",
            max_pending_images=2,
        )
        == 1
    )
    prefetcher.wait(timeout=5)
    assert prefetcher._accepted_images == {"image:a", "image:b", "image:c"}
    prefetcher.close()


def test_dataset_submission_does_not_wait_for_prefetch(monkeypatch):
    _enable(monkeypatch)
    prefetcher = _SWEValidationImagePrefetcher()
    started = threading.Event()
    release = threading.Event()

    def blocked_prefetch(images, _batch_index, _label):
        started.set()
        assert release.wait(timeout=5)
        return {"ok": True, "images": images}

    monkeypatch.setattr(prefetcher, "_prefetch", blocked_prefetch)

    assert prefetcher.submit_dataset(_Loader(["image:a"]), start_index=0, label="validation") == 1
    assert started.wait(timeout=2)
    with prefetcher._lock:
        assert any(not future.done() for future in prefetcher._futures)

    release.set()
    prefetcher.wait(timeout=5)
    prefetcher.close()


def test_prefetch_queue_deduplicates_images_and_chunks_requests(monkeypatch):
    _enable(monkeypatch, chunk_size=2, parallelism=50)
    prefetcher = _SWEValidationImagePrefetcher()
    chunks = []

    def record_prefetch(images, _batch_index, label):
        chunks.append((images, label))
        return {"ok": True, "images": images}

    monkeypatch.setattr(prefetcher, "_prefetch", record_prefetch)

    assert prefetcher.submit_dataset(
        _Loader(["image:a", "image:b", "image:a", "image:c"]),
        start_index=0,
        label="validation",
    ) == 3
    prefetcher.wait(timeout=5)

    assert [images for images, _label in chunks] == [["image:a", "image:b"], ["image:c"]]
    prefetcher.close()


def test_dataset_prefetch_respects_resume_offset(monkeypatch):
    _enable(monkeypatch)
    prefetcher = _SWEValidationImagePrefetcher()
    pulled = []

    def record_prefetch(images, _batch_index, _label):
        pulled.extend(images)
        return {"ok": True, "images": images}

    monkeypatch.setattr(prefetcher, "_prefetch", record_prefetch)

    assert prefetcher.submit_dataset(
        _Loader(["image:done", "image:next", "image:last"]),
        start_index=1,
        label="validation",
    ) == 2
    prefetcher.wait(timeout=5)

    assert pulled == ["image:next", "image:last"]
    prefetcher.close()


def test_prefetch_capacity_response_retries_before_marking_image_accepted(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setenv("SWE_AGENT_VALIDATION_IMAGE_PREFETCH_RETRY_INITIAL_SECONDS", "0.01")
    prefetcher = _SWEValidationImagePrefetcher()
    calls = 0

    def capacity_then_accept(images, _batch_index, _label):
        nonlocal calls
        calls += 1
        if calls == 1:
            http_error = urllib.error.HTTPError(
                "http://execution.test:18080/prefetch_images",
                429,
                "capacity",
                {"Retry-After": "0.01"},
                io.BytesIO(b'{"error_code":"image_prefetch_capacity_exhausted"}'),
            )
            raise RuntimeError("HTTP 429: image prefetch queue full") from http_error
        return {"ok": True, "images": images}

    monkeypatch.setattr(prefetcher, "_prefetch", capacity_then_accept)
    monkeypatch.setattr(ray_trainer.time, "sleep", lambda _seconds: None)

    assert prefetcher.submit_images(["image:a"], batch_index=1, label="batch1") == 1
    prefetcher.wait(timeout=5)

    assert calls == 2
    assert prefetcher._accepted_images == {"image:a"}
    assert not prefetcher._pending_images
    assert prefetcher.submit_images(["image:a"], batch_index=2, label="batch2") == 0
    prefetcher.close()


def test_terminal_prefetch_failure_allows_later_resubmission(monkeypatch):
    _enable(monkeypatch)
    prefetcher = _SWEValidationImagePrefetcher()
    calls = 0

    def fail_then_accept(images, _batch_index, _label):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("invalid prefetch request")
        return {"ok": True, "images": images}

    monkeypatch.setattr(prefetcher, "_prefetch", fail_then_accept)

    assert prefetcher.submit_images(["image:a"], batch_index=1, label="batch1") == 1
    prefetcher.wait(timeout=5)
    assert not prefetcher._pending_images
    assert not prefetcher._accepted_images

    assert prefetcher.submit_images(["image:a"], batch_index=2, label="batch2") == 1
    prefetcher.wait(timeout=5)
    assert calls == 2
    assert prefetcher._accepted_images == {"image:a"}
    prefetcher.close()


def test_advancing_batch_cancels_stale_queued_chunks(monkeypatch):
    _enable(monkeypatch, chunk_size=1)
    prefetcher = _SWEValidationImagePrefetcher()
    started = threading.Event()
    release = threading.Event()
    pulled = []

    def block_first_chunk(images, _batch_index, _label):
        pulled.extend(images)
        started.set()
        assert release.wait(timeout=5)
        return {"ok": True, "images": images}

    monkeypatch.setattr(prefetcher, "_prefetch", block_first_chunk)

    prefetcher.advance_batch(1)
    assert prefetcher.submit_images(["image:a", "image:b"], batch_index=1, label="batch1") == 2
    assert started.wait(timeout=2)

    prefetcher.advance_batch(2)
    release.set()
    prefetcher.wait(timeout=5)

    assert pulled == ["image:a"]
    assert not prefetcher._pending_images
    assert prefetcher._accepted_images == {"image:a"}
    prefetcher.close()


def test_multiservice_prefetch_and_claim_share_deterministic_image_owner(monkeypatch):
    urls = ["http://a:18080", "http://b:18080", "http://c:18080"]
    monkeypatch.setenv("SWE_AGENT_VALIDATION_IMAGE_PREFETCH", "1")
    monkeypatch.setenv("SWE_AGENT_EXECUTION_URLS", ",".join(urls))
    monkeypatch.setenv("SWE_AGENT_IMAGE_PREFETCH_REPLICATION_FACTOR", "2")
    trainer_router = ExecutionServerRouter(urls)
    endpoint_calls = {}

    def record_post(path, payload, config):
        assert path == "/prefetch_images"
        url = config["execution_url_override"]
        endpoint_calls[url] = list(payload["images"])
        return {"ok": True, "accepted": True, "images": payload["images"]}

    monkeypatch.setattr(ray_trainer, "get_execution_router", lambda: trainer_router)
    monkeypatch.setattr(ray_trainer, "_remote_post_json", record_post)
    prefetcher = _SWEValidationImagePrefetcher()
    images = ["benchmark:image-a", "benchmark:image-b", "benchmark:image-c"]

    result = prefetcher._prefetch(images, batch_index=1, label="validation")

    assert result["ok"] is True
    assert result["replication_factor"] == 2
    assert endpoint_calls == trainer_router.partition_images(images, replication_factor=2)

    worker_router = ExecutionServerRouter(list(reversed(urls)))
    monkeypatch.setattr(worker_router, "_refresh", lambda: None)
    for state in worker_router._states.values():
        state.healthy = True
    for index, image in enumerate(images):
        assert worker_router.select(f"request-{index}", image=image, resource="/claim") == (
            trainer_router.image_affinity_order(image)[0]
        )
    prefetcher.close()
