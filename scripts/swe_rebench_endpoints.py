"""Endpoint availability is separate from immutable reference evidence."""

from __future__ import annotations

import collections
import copy
import heapq
import threading
import time

BACKOFF = (5, 15, 30, 60)
IDLE_TIMEOUT = 600
READY_PROBE_INTERVAL = 30
TASK_FAILOVER_LIMIT = 3
MONITOR_INTERVAL = .5


class PendingTasks:
    """Indexed dispatch/recovery queues. Caller owns the scheduling condition."""
    def __init__(self, states, urls, endpoints, holds, uncertain, needs_cleanup, failures):
        self.states, self.urls, self.endpoints = states, set(urls), endpoints
        self.holds, self.uncertain, self.needs_cleanup, self.failures = holds, uncertain, needs_cleanup, failures
        self.items = {}
        self.queues = collections.defaultdict(collections.deque)
        self.generations = collections.Counter()
        self.delayed = []

    def __bool__(self):
        return bool(self.items)

    def __iter__(self):
        return iter(self.items.values())

    def bucket(self, key):
        state = self.states[key]
        if state.get("terminal"):
            return ("terminal", None)
        if self.failures[key] >= TASK_FAILOVER_LIMIT:
            return ("parked", None)
        owner = state.get("endpoint")
        if key in self.uncertain:
            return ("recovery", owner)
        if (owner and self.needs_cleanup.get(key) and not state.get("migration_pending")
                and not self.endpoints.cleanup_ready(owner)):
            return ("parked", None)
        if (owner in self.urls and self.endpoints.ready(owner)
                and not state.get("restart_on_failover") and not state.get("migration_pending")):
            return ("owner", owner)
        return ("free", None)

    def put(self, case, delay=0):
        key = case["inventory"]["id"]
        if key in self.items:
            raise ValueError("duplicate pending task")
        self.items[key] = case
        self.generations[key] += 1
        entry = (key, self.generations[key])
        if delay:
            heapq.heappush(self.delayed, (time.monotonic() + delay, *entry))
        else:
            self.queues[self.bucket(key)].append(entry)

    def _due(self):
        now = time.monotonic()
        while self.delayed and self.delayed[0][0] <= now:
            _, key, generation = heapq.heappop(self.delayed)
            if key in self.items and self.generations[key] == generation:
                self.queues[self.bucket(key)].append((key, generation))

    def refresh(self):
        # Reindex only on endpoint availability transitions, never per worker
        # wakeup. Delayed entries remain subject to their original backoff.
        delayed = {key for _, key, generation in self.delayed
                   if self.generations[key] == generation and key in self.items}
        self.queues.clear()
        for key in self.items:
            if key not in delayed:
                self.queues[self.bucket(key)].append((key, self.generations[key]))

    def take(self, url, *, recovery=False):
        self._due()
        if recovery:
            buckets = [("recovery", url)] if self.endpoints.cleanup_ready(url) else []
        else:
            buckets = [("terminal", None)]
            if self.endpoints.ready(url) and not self.holds[url]:
                buckets += [("owner", url), ("free", None)]
        for bucket in buckets:
            queue = self.queues[bucket]
            while queue:
                key, generation = queue.popleft()
                if key not in self.items or self.generations[key] != generation:
                    continue
                actual = self.bucket(key)
                if actual != bucket:
                    self.queues[actual].append((key, generation))
                    continue
                return self.items.pop(key)
        return None


class EndpointUnavailable(RuntimeError):
    pass


class DeploymentMismatch(EndpointUnavailable):
    pass


class Endpoints:
    def __init__(self, client, urls, source):
        self.client, self.urls, self.source = client, urls, source
        self.lock = threading.Lock()
        self.states = {url: {"ready": False, "cleanup_ready": False, "failures": 0, "next_probe": 0}
                       for url in urls}

    def add_cleanup_endpoint(self, url):
        """Historical owners remain observable for cleanup, never new dispatch."""
        with self.lock:
            if url not in self.states:
                self.states[url] = {"ready": False, "cleanup_ready": False, "failures": 0, "next_probe": 0}

    def deployment(self, url):
        try:
            value = self.client.get(url, "/tool_contract_info?summary=1")
        except Exception as exc:
            raise EndpointUnavailable("deployment probe unavailable: " + url) from exc
        if not value.get("source_fingerprint", {}).get("sha256"):
            raise EndpointUnavailable("deployment probe unavailable: " + url + ": " + str(value)[:600])
        if value["source_fingerprint"] != self.source or value.get("audit_image_lifecycle_version") != 1:
            raise DeploymentMismatch("matching services and audit image lifecycle v1 required: " + url)
        return {"url": url, "source_sha256": value["source_fingerprint"]["sha256"],
                "checked_at_unix": time.time()}

    def check(self, url, *, admission=True):
        evidence = self.deployment(url)
        try:
            health = self.client.get(url, "/status")
        except Exception as exc:
            raise EndpointUnavailable("status probe unavailable: " + url) from exc
        if health.get("ok") is not True or health.get("docker_available") is not True:
            raise EndpointUnavailable("Docker/service unavailable: " + url + ": " + str(health)[:600])
        if admission and health.get("workspace_admission", {}).get("ok") is not True:
            raise EndpointUnavailable("workspace capacity unavailable: " + url + ": " + str(health)[:600])
        return evidence

    def refresh(self, url):
        try:
            evidence = self.deployment(url)
            health = self.client.get(url, "/status")
            cleanup_ready = health.get("ok") is True and health.get("docker_available") is True
            ready = cleanup_ready and health.get("workspace_admission", {}).get("ok") is True
            error = None if ready else str(health)[:1000]
            observation = {"deployment": evidence, "health": health, "cleanup_ready": cleanup_ready,
                           "ready": ready, "error": error, "deployment_mismatch": False}
        except Exception as exc:
            observation = {"ready": False, "cleanup_ready": False, "error": str(exc)[:1000],
                           "deployment_mismatch": isinstance(exc, DeploymentMismatch)}
        with self.lock:
            previous = self.states[url]
            failures = 0 if observation["ready"] else previous["failures"] + 1
            delay = READY_PROBE_INTERVAL if observation["ready"] else BACKOFF[min(failures - 1, len(BACKOFF) - 1)]
            self.states[url] = {**observation, "failures": failures, "next_probe": time.monotonic() + delay,
                                "updated_at_unix": time.time()}
            return copy.deepcopy(self.states[url])

    def pause(self, url, reason):
        with self.lock:
            state = self.states[url]
            failures = state["failures"] + 1
            state.update(ready=False, error=str(reason)[:1000], failures=failures,
                         next_probe=time.monotonic() + BACKOFF[min(failures - 1, len(BACKOFF) - 1)],
                         updated_at_unix=time.time())

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.states)

    def ready(self, url):
        with self.lock:
            return self.states[url]["ready"]

    def cleanup_ready(self, url):
        with self.lock:
            return self.states[url]["cleanup_ready"]

    def due(self):
        with self.lock:
            now = time.monotonic()
            return [url for url, state in self.states.items() if state["next_probe"] <= now]
