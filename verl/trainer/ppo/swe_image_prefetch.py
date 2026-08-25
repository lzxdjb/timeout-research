"""Trainer adapters for nonblocking SWE Docker image prefetching.

The SWE implementation remains in the stock-rl-reflect checkout. This module
only connects its execution router and task payloads to the target trainer's
batch lifecycle.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np

try:
    from recipe.swe_agent.execution_router import configured_execution_urls, get_execution_router
    from recipe.swe_agent.task_payload import rollout_task_payload
    from recipe.swe_agent.tools import _remote_post_json
except ModuleNotFoundError as exc:
    if not str(exc.name or "").startswith("recipe.swe_agent"):
        raise
    _SWE_IMPORT_ERROR = exc

    def configured_execution_urls() -> list[str]:
        return []

    def _missing_swe_source(*_args, **_kwargs):
        raise RuntimeError(
            "SWE image prefetch requires the stock-rl-reflect checkout on PYTHONPATH"
        ) from _SWE_IMPORT_ERROR

    get_execution_router = _missing_swe_source
    rollout_task_payload = _missing_swe_source
    _remote_post_json = _missing_swe_source

def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_swe_task(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _swe_task_from_tools_kwargs(tools_kwargs: Any) -> dict[str, Any]:
    if not isinstance(tools_kwargs, dict):
        return {}
    task = _load_swe_task(tools_kwargs.get("__swe_task_json__"))
    if task:
        return task
    for value in tools_kwargs.values():
        if not isinstance(value, dict):
            continue
        create_kwargs = value.get("create_kwargs") if isinstance(value.get("create_kwargs"), dict) else {}
        task = _load_swe_task(create_kwargs.get("task_json"))
        if task:
            return task
    return {}


def _iter_object_array(value: Any):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list | tuple):
        return list(value)
    return [value]
class _SWEImagePrefetcher:
    """Keep a bounded batch-image lookahead supplied in the background."""

    def __init__(self, *, scope: str, log_name: str, source: str):
        self.scope = scope.upper()
        self.log_name = log_name
        self.source = source
        env_prefix = f"SWE_AGENT_{self.scope}_IMAGE_PREFETCH"
        self.enabled = _env_flag(env_prefix, False)
        if self.enabled and "_SWE_IMPORT_ERROR" in globals():
            _missing_swe_source()
        self.strict = _env_flag(f"{env_prefix}_STRICT", False)
        self.execution_urls = configured_execution_urls()
        self.execution_url = self.execution_urls[0] if self.execution_urls else ""
        self.token = os.environ.get("SWE_AGENT_EXECUTION_TOKEN", "")
        self.timeout = _env_float(
            f"{env_prefix}_TIMEOUT",
            _env_float("SWE_AGENT_EXECUTION_HTTP_TIMEOUT", 7200.0),
        )
        self.parallelism = max(
            1,
            _env_int(f"{env_prefix}_PARALLELISM", 2),
        )
        self.chunk_size = max(
            1,
            _env_int(f"{env_prefix}_CHUNK_SIZE", 16),
        )
        self.replication_factor = max(
            1,
            _env_int("SWE_AGENT_IMAGE_PREFETCH_REPLICATION_FACTOR", 1),
        )
        self.lookahead_batches = min(
            1,
            max(0, _env_int(f"{env_prefix}_LOOKAHEAD_BATCHES", 1)),
        )
        self.retry_timeout = max(
            0.0,
            _env_float(f"{env_prefix}_RETRY_TIMEOUT_SECONDS", self.timeout),
        )
        self.retry_initial = max(
            0.01,
            _env_float(f"{env_prefix}_RETRY_INITIAL_SECONDS", 1.0),
        )
        self.retry_max = max(
            self.retry_initial,
            _env_float(f"{env_prefix}_RETRY_MAX_SECONDS", 30.0),
        )
        self._queue: queue.Queue[tuple[concurrent.futures.Future, list[str], int, str] | None] | None = None
        self._worker: threading.Thread | None = None
        self._futures: list[concurrent.futures.Future] = []
        self._pending_images: set[str] = set()
        self._accepted_images: set[str] = set()
        self._errors: list[BaseException] = []
        self._lock = threading.RLock()
        self._active_batch_index = 0
        self._closed = False
        if self.enabled and not self.execution_urls:
            raise RuntimeError(
                f"{env_prefix}=1 requires SWE_AGENT_EXECUTION_URL or "
                "SWE_AGENT_EXECUTION_URLS"
            )
        if self.enabled:
            self._queue = queue.Queue()
            self._worker = threading.Thread(
                target=self._worker_loop,
                name=f"swe-{log_name}-image-prefetch",
                daemon=True,
            )
            self._worker.start()
            print(
                f"[{self.log_name}_prefetch] enabled mode=image_only_nonblocking "
                f"current_batch=true lookahead_batches={self.lookahead_batches} "
                f"chunk_size={self.chunk_size} requested_parallelism={self.parallelism}",
                flush=True,
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._queue is not None:
            self._queue.put(None)

    def submit_batch(self, batch_dict: dict[str, Any], *, batch_index: int, label: str) -> int:
        return self.submit_images(
            self._images_from_batch(batch_dict),
            batch_index=batch_index,
            label=label,
        )

    def submit_batch_bounded(
        self,
        batch_dict: dict[str, Any],
        *,
        batch_index: int,
        label: str,
        max_pending_images: int,
    ) -> int:
        """Submit without letting fully-async producers build an unbounded queue."""
        return self.submit_images(
            self._images_from_batch(batch_dict),
            batch_index=batch_index,
            label=label,
            max_pending_images=max_pending_images,
        )

    def advance_batch(self, batch_index: int) -> None:
        """Discard client work that can no longer help the active batch."""
        if not self.enabled:
            return
        with self._lock:
            self._active_batch_index = max(self._active_batch_index, batch_index)
            stale = [
                future
                for future in self._futures
                if getattr(future, "_swe_prefetch_batch_index", batch_index) < batch_index
                and not future.running()
                and not future.done()
            ]
        for future in stale:
            future.cancel()

    def submit_dataset(self, val_dataloader, *, start_index: int, label: str) -> int:
        return self.submit_images(
            self._images_from_dataset(val_dataloader, start_index=start_index),
            batch_index=0,
            label=label,
        )

    def submit_images(
        self,
        images: list[str],
        *,
        batch_index: int,
        label: str,
        max_pending_images: int | None = None,
    ) -> int:
        if not self.enabled or self._queue is None or self._closed:
            return 0
        queued: list[str] = []
        skipped_for_capacity = 0
        with self._lock:
            if self._active_batch_index and batch_index > self._active_batch_index + self.lookahead_batches:
                return 0
            available = None
            if max_pending_images is not None:
                available = max(0, int(max_pending_images) - len(self._pending_images))
            for image in images:
                if (
                    image
                    and image not in self._accepted_images
                    and image not in self._pending_images
                ):
                    if available is not None and len(queued) >= available:
                        skipped_for_capacity += 1
                        continue
                    self._pending_images.add(image)
                    queued.append(image)
            for offset in range(0, len(queued), self.chunk_size):
                chunk = queued[offset : offset + self.chunk_size]
                chunk_label = f"{label}:{offset + 1}-{offset + len(chunk)}"
                future: concurrent.futures.Future = concurrent.futures.Future()
                future._swe_prefetch_images = tuple(chunk)
                future._swe_prefetch_batch_index = batch_index
                future.add_done_callback(self._record_completion)
                self._futures.append(future)
                self._queue.put((future, chunk, batch_index, chunk_label))
        if skipped_for_capacity:
            print(
                f"[{self.log_name}_prefetch] capacity_full skipped_images={skipped_for_capacity} "
                f"max_pending_images={max_pending_images} lazy_claim_fallback=true",
                flush=True,
            )
        if queued:
            print(
                f"[{self.log_name}_prefetch] queued label={label} images={len(queued)} "
                f"chunk_size={self.chunk_size} requested_parallelism={self.parallelism} "
                f"replication_factor={self.replication_factor}",
                flush=True,
            )
        return len(queued)

    @staticmethod
    def _images_from_batch(batch_dict: dict[str, Any]) -> list[str]:
        tools_kwargs_batch = _iter_object_array(batch_dict.get("tools_kwargs"))
        images: list[str] = []
        for tools_kwargs in tools_kwargs_batch:
            task = _swe_task_from_tools_kwargs(tools_kwargs)
            if not task:
                continue
            task = rollout_task_payload(task)
            environment = task.get("environment") if isinstance(task.get("environment"), dict) else {}
            image = str(task.get("docker_image") or environment.get("image_tag") or "").strip()
            if image and image not in images:
                images.append(image)
        return images

    @classmethod
    def _images_from_dataset(cls, val_dataloader, *, start_index: int = 0) -> list[str]:
        dataset = getattr(val_dataloader, "dataset", None)
        dataframe = getattr(dataset, "dataframe", None)
        column_names = set(getattr(dataframe, "column_names", ()))
        if dataframe is None or "extra_info" not in column_names:
            return []
        try:
            extra_infos = dataframe["extra_info"]
        except (KeyError, TypeError, ValueError):
            return []
        images: list[str] = []
        for extra_info in extra_infos[max(0, start_index) :]:
            if not isinstance(extra_info, dict):
                continue
            batch = {"tools_kwargs": [extra_info.get("tools_kwargs", {})]}
            for image in cls._images_from_batch(batch):
                if image not in images:
                    images.append(image)
        return images

    def _worker_loop(self) -> None:
        work_queue = self._queue
        assert work_queue is not None
        while True:
            item = work_queue.get()
            try:
                if item is None:
                    return
                future, images, batch_index, label = item
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    response = self._prefetch_with_retry(images, batch_index, label)
                except BaseException as exc:
                    future.set_exception(exc)
                else:
                    future.set_result(response)
            finally:
                work_queue.task_done()

    def _record_completion(self, future: concurrent.futures.Future) -> None:
        try:
            response = future.result()
        except concurrent.futures.CancelledError:
            with self._lock:
                images = getattr(future, "_swe_prefetch_images", ())
                self._pending_images.difference_update(images)
            return
        except BaseException as exc:
            with self._lock:
                images = getattr(future, "_swe_prefetch_images", ())
                self._pending_images.difference_update(images)
                self._errors.append(exc)
            if not self.strict:
                print(
                    f"[{self.log_name}_prefetch] failed_lazy_claim_fallback error={exc}",
                    flush=True,
                )
            return
        images = response.get("images", [])
        skipped_images = response.get("skipped_images", [])
        with self._lock:
            self._pending_images.difference_update(images)
            self._pending_images.difference_update(skipped_images)
            self._accepted_images.update(images)
        print(
            f"[{self.log_name}_prefetch] accepted images={len(images)}",
            flush=True,
        )

    @staticmethod
    def _retryable_prefetch_error(exc: BaseException) -> bool:
        current: BaseException | None = exc
        while current is not None:
            if isinstance(current, urllib.error.HTTPError):
                return current.code in {429, 502, 503, 504}
            if isinstance(current, OSError):
                return True
            current = current.__cause__
        return "HTTP 429:" in str(exc)

    @staticmethod
    def _prefetch_retry_after(exc: BaseException) -> float:
        current: BaseException | None = exc
        while current is not None:
            if isinstance(current, urllib.error.HTTPError):
                raw = current.headers.get("Retry-After") if current.headers else None
                try:
                    return max(0.0, float(raw)) if raw is not None else 0.0
                except (TypeError, ValueError):
                    return 0.0
            current = current.__cause__
        return 0.0

    def _prefetch_with_retry(self, images: list[str], batch_index: int, label: str) -> dict[str, Any]:
        started = time.monotonic()
        deadline = started + self.retry_timeout
        delay = self.retry_initial
        attempts = 0
        while True:
            attempts += 1
            with self._lock:
                stale = batch_index < self._active_batch_index
            if stale:
                return {"ok": True, "images": [], "skipped_images": images, "stale": True}
            try:
                response = self._prefetch(images, batch_index, label)
                if not response.get("ok"):
                    raise RuntimeError(f"prefetch was not accepted: {response}")
                if attempts > 1:
                    print(
                        f"[{self.log_name}_prefetch] retry completed label={label} attempts={attempts} "
                        f"elapsed_seconds={time.monotonic() - started:.2f}",
                        flush=True,
                    )
                return response
            except BaseException as exc:
                remaining = deadline - time.monotonic()
                if (
                    not self._retryable_prefetch_error(exc)
                    or self.retry_timeout <= 0
                    or remaining <= 0
                ):
                    raise
                sleep_seconds = min(remaining, max(delay, self._prefetch_retry_after(exc)))
                if attempts == 1 or attempts % 20 == 0:
                    print(
                        f"[{self.log_name}_prefetch] retrying label={label} attempts={attempts} "
                        f"sleep_seconds={sleep_seconds:.2f} error={str(exc)[:500]}",
                        flush=True,
                    )
                time.sleep(sleep_seconds)
                delay = min(self.retry_max, max(self.retry_initial, delay * 2.0))

    def raise_completed_errors(self) -> None:
        if not self.strict:
            return
        with self._lock:
            error = self._errors[0] if self._errors else None
        if error is not None:
            raise RuntimeError(f"SWE {self.log_name} image prefetch failed: {error}") from error

    def wait(self, *, timeout: float | None = None) -> None:
        with self._lock:
            futures = list(self._futures)
        if futures:
            _done, pending = concurrent.futures.wait(futures, timeout=timeout)
            if pending:
                raise TimeoutError(
                    f"Timed out with {len(pending)} {self.log_name} image prefetch chunks pending"
                )
        self.raise_completed_errors()

    def _prefetch(self, images: list[str], batch_index: int, label: str) -> dict[str, Any]:
        payload = {
            "images": images,
            "parallelism": self.parallelism,
            "source": self.source,
            "summary": {"batch_index": batch_index, "label": label, "images": len(images)},
        }
        router = get_execution_router()
        if router is not None:
            endpoint_results: list[dict[str, Any]] = []

            def prefetch_endpoint(item: tuple[str, list[str]]) -> dict[str, Any]:
                url, endpoint_images = item
                endpoint_payload = dict(payload)
                endpoint_payload["images"] = endpoint_images
                return _remote_post_json(
                    "/prefetch_images",
                    endpoint_payload,
                    {"execution_url_override": url, "execution_token": self.token},
                )

            if router.image_affinity_enabled:
                buckets = router.partition_images(
                    images,
                    replication_factor=self.replication_factor,
                )
                effective_replication_factor = min(
                    self.replication_factor,
                    len(self.execution_urls),
                )
            else:
                buckets = router.partition(images, resource="/prefetch_images")
                effective_replication_factor = 1
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(buckets), thread_name_prefix="swe-image-prefetch-endpoint"
            ) as executor:
                futures = {
                    executor.submit(prefetch_endpoint, item): item
                    for item in buckets.items()
                }
                for future in concurrent.futures.as_completed(futures):
                    url, endpoint_images = futures[future]
                    result = future.result()
                    result = dict(result)
                    result["execution_url"] = url
                    result["assigned_images"] = endpoint_images
                    endpoint_results.append(result)
                    if result.get("ok") and not router.image_affinity_enabled:
                        for image in endpoint_images:
                            router.bind_image(image, url)
            return {
                "ok": all(bool(result.get("ok")) for result in endpoint_results),
                "images": images,
                "image_affinity_enabled": router.image_affinity_enabled,
                "replication_factor": effective_replication_factor,
                "endpoint_results": endpoint_results,
            }

        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(
            f"{self.execution_url}/prefetch_images",
            data=raw,
            headers=headers,
            method="POST",
        )
        # Internal pod-to-pod traffic must bypass environment proxy variables.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body[:2000]}") from exc
        parsed = json.loads(body or "{}")
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Expected JSON object from execution service, got: {body[:1000]}")
        return parsed


class _SWEValidationImagePrefetcher(_SWEImagePrefetcher):
    def __init__(self):
        super().__init__(
            scope="VALIDATION",
            log_name="validate",
            source="verl.validation.image_prefetch",
        )


class _SWETrainingImagePrefetcher(_SWEImagePrefetcher):
    def __init__(self):
        super().__init__(
            scope="TRAINING",
            log_name="training",
            source="verl.training.image_prefetch",
        )


def _validate_swe_training_prefetch_modes() -> None:
    if _env_flag("SWE_AGENT_TRAINER_INTEGRATED_PREWARM", False):
        raise ValueError(
            "SWE_AGENT_TRAINER_INTEGRATED_PREWARM is not supported by this integration; "
            "use nonblocking SWE_AGENT_TRAINING_IMAGE_PREFETCH instead."
        )
