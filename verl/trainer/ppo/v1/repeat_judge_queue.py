"""Opt-in repeat-only inference coordinator for V1 synchronous rollouts."""

import asyncio
import logging
import math
import os
import time

import ray


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _debug_bool(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).lower()
    if value not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
        raise ValueError(f"{name} must be a boolean")
    return value in {"1", "true", "yes", "on"}


def _debug_slow_seconds() -> float:
    value = float(os.getenv("SWE_AGENT_REPEAT_JUDGE_DEBUG_SLOW_SECONDS", "30"))
    if value < 0:
        raise ValueError("SWE_AGENT_REPEAT_JUDGE_DEBUG_SLOW_SECONDS must be nonnegative")
    return value


def progress_interval_seconds() -> float:
    value = float(os.getenv("SWE_AGENT_REPEAT_JUDGE_PROGRESS_INTERVAL_SECONDS", "60"))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("SWE_AGENT_REPEAT_JUDGE_PROGRESS_INTERVAL_SECONDS must be finite and positive")
    return value


class RepeatQueueError(RuntimeError):
    """A lost/mismatched queued judgment must abort the step."""


def deferred_enabled() -> bool:
    value = os.getenv("SWE_AGENT_REPEAT_JUDGE_DEFERRED_ENABLED", "0").lower()
    if value not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
        raise ValueError("SWE_AGENT_REPEAT_JUDGE_DEFERRED_ENABLED must be a boolean")
    return value in {"1", "true", "yes", "on"}


def validate_deferred_config(config):
    if deferred_enabled() and (
        not config.trainer.get("use_v1", False) or config.trainer.v1.trainer_mode != "sync"
    ):
        raise ValueError("Deferred repeat judging requires trainer.use_v1=True and trainer.v1.trainer_mode=sync")
    if deferred_enabled() and config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class"):
        raise ValueError("Deferred repeat judging currently requires the built-in AgentLoopManagerTQ")


def finalize_output(output, score, metadata):
    """Merge once, keeping hidden/raw scores and every shaped-score copy consistent."""
    if output.extra_fields.get("_swe_repeat_finalized"):
        raise RuntimeError("Repeat reward was finalized twice")
    extra = output.extra_fields
    info = extra["reward_extra_info"]
    info.update(metadata, score=score, shaped_score=score)
    if info.get("noninfra_shaped_score") is not None:
        info["noninfra_shaped_score"] = score
    hidden = extra.get("swe_hidden_reward")
    if hidden is not None:
        hidden["shaped_score"] = score
        if hidden.get("ok"):
            hidden["metadata"].update(metadata, shaped_score=score)
            if hidden["metadata"].get("noninfra_shaped_score") is not None:
                hidden["metadata"]["noninfra_shaped_score"] = score
    output.reward_score = score
    extra["_swe_repeat_finalized"] = True


class RepeatJudgeCoordinator:
    """Bound accepted jobs without waiting for capacity in generation sessions.

    Queue-full submissions preserve the base reward with an uncertain verdict.
    Actor failures still propagate. The shared client limits HTTP calls, while
    job_slots keeps a bounded number of whole vote sequences active at once.
    """

    def __init__(self, config):
        self.config = config
        self.client = None
        self.job_slots = asyncio.Semaphore(config.concurrency)
        self.pending = set()
        self.running = set()
        self.counts = dict(submitted=0, accepted=0, completed=0, failed=0, cancelled=0, dropped=0)
        self.debug = _debug_bool("SWE_AGENT_REPEAT_JUDGE_DEBUG")
        self.debug_slow_seconds = _debug_slow_seconds()
        self._last_debug_snapshot = 0.0
        if self.debug:
            logger.info(
                "Repeat judge debug startup endpoint=%s model=%s max_context=%s max_output=%s "
                "timeout=%s queue_timeout=%s concurrency=%s vote_count=%s vote_quorum=%s max_pending=%s "
                "early_exit_enabled=%s",
                config.url, config.model, config.max_context, config.max_output,
                config.timeout, config.queue_timeout, config.concurrency,
                config.vote_count, config.vote_quorum, config.max_pending, config.early_exit_enabled,
            )

    def snapshot(self):
        """Payload-free counters, also available for live actor diagnostics."""
        return {**self.counts, "queued": len(self.pending - self.running),
                "active": len(self.running), "pending": len(self.pending),
                "capacity": self.config.max_pending - len(self.pending)}

    def _debug_snapshot(self, reason: str, force: bool = False) -> None:
        if not self.debug:
            return
        now = time.monotonic()
        interval = max(1.0, min(self.debug_slow_seconds or 1.0, 30.0))
        if not force and now - self._last_debug_snapshot < interval:
            return
        self._last_debug_snapshot = now
        logger.info("Repeat judge queue snapshot reason=%s %s", reason,
                    " ".join(f"{key}={value}" for key, value in self.snapshot().items()))

    def _debug_duration(self, phase: str, job_id: str, started: float) -> None:
        if not self.debug:
            return
        duration = time.monotonic() - started
        level = logging.WARNING if duration >= self.debug_slow_seconds else logging.INFO
        logger.log(level, "Repeat judge debug phase=%s job_id=%s seconds=%.3f slow=%s",
                   phase, job_id, duration, duration >= self.debug_slow_seconds)

    async def ready(self):
        from recipe.swe_agent import repetition_reward as repeat
        self.client = repeat.get_client(self.config)
        if self.debug:
            logger.info("Repeat judge debug preflight start endpoint=%s", self.config.url)
        started = time.monotonic()
        await self.client.preflight()
        if self.debug:
            self._debug_duration("preflight", "startup", started)
            self._debug_snapshot("ready", force=True)

    async def submit(self, job_id, job):
        """Admit immediately or skip; the returned result is awaited by a finalizer."""
        from recipe.swe_agent import repetition_reward as repeat

        started = time.monotonic()
        if job["config"] != self.config:
            raise RuntimeError("Repeat coordinator configuration mismatch")
        if job_id in self.pending:
            raise RuntimeError("Duplicate repeat job: " + job_id)
        self.counts["submitted"] += 1
        job = {**job, "metadata": dict(job["metadata"])}
        job["metadata"].update(repeat_reward_pending=0, repeat_reward_deferred=1)
        # No await between the capacity check and registration: async actor
        # submissions cannot race admission or create a producer-side waiter.
        if len(self.pending) >= self.config.max_pending:
            self.counts["dropped"] += 1
            job["metadata"].update(repeat_reward_queue_full=1, repeat_reward_queue_accepted=0,
                                   repeat_reward_judge_called=0, repeat_reward_stage=0,
                                   repeat_reward_deferred_admission_seconds=0.0,
                                   repeat_reward_deferred_job_queue_seconds=0.0)
            self._debug_snapshot("queue_full", force=True)
            score, metadata = await repeat.finish_repeat_reward(job, repeat.uncertain("repeat_queue_full"))
            metadata["repeat_reward_deferred_wall_seconds"] = time.monotonic() - started
            return job_id, score, metadata

        self.pending.add(job_id)
        self.counts["accepted"] += 1
        admission_seconds = time.monotonic() - started
        job["metadata"].update(repeat_reward_queue_full=0, repeat_reward_queue_accepted=1)
        self._debug_snapshot("submitted")
        acquired = False
        try:
            await self.job_slots.acquire()
            acquired = True
            self.running.add(job_id)
            queue_seconds = time.monotonic() - started
            self._debug_snapshot("job_started")
            actor = job["actor_segments"]
            judge_started = time.monotonic()
            if self.debug:
                logger.info("Repeat judge RPC submitted job_id=%s", job_id)
            verdict = await self.client.judge_consensus(
                job["transcript"], job["events"], **({"actor_segments": actor} if actor is not None else {})
            )
            if self.debug:
                self._debug_duration("judge_rpc", job_id, judge_started)
                logger.info("Repeat judge verdict job_id=%s verdict=%s queue_seconds=%.3f service_seconds=%.3f",
                            job_id, verdict.get("verdict", ""), float(verdict.get("queue_seconds", 0.0)),
                            float(verdict.get("service_seconds", 0.0)))
            job["metadata"].update(
                repeat_reward_judge_called=1, repeat_reward_stage=2,
                repeat_reward_pending=0, repeat_reward_deferred=1,
            )
            score, metadata = await repeat.finish_repeat_reward(job, verdict)
            metadata.update(
                repeat_reward_deferred_admission_seconds=admission_seconds,
                repeat_reward_deferred_job_queue_seconds=queue_seconds,
                repeat_reward_deferred_wall_seconds=time.monotonic() - started,
                repeat_reward_deferred_request_queue_seconds=verdict.get("queue_seconds", 0.0),
                repeat_reward_deferred_service_seconds=verdict.get("service_seconds", 0.0),
            )
            self.counts["completed"] += 1
            return job_id, score, metadata
        except asyncio.CancelledError:
            self.counts["cancelled"] += 1
            raise
        except Exception as error:
            self.counts["failed"] += 1
            if self.debug:
                logger.warning("Repeat judge evaluate error job_id=%s type=%s message=%s",
                               job_id, type(error).__name__, str(error)[:240])
            raise
        finally:
            self.running.discard(job_id)
            self.pending.remove(job_id)
            if acquired:
                self.job_slots.release()
            if self.debug:
                self._debug_duration("evaluate_total", job_id, started)
                self._debug_snapshot("evaluation_released", force=True)


# No GPUs: inference remains on the configured external endpoint. No automatic replay of lost jobs.
RepeatJudgeCoordinatorActor = ray.remote(num_cpus=1, max_concurrency=10000, max_restarts=0)(RepeatJudgeCoordinator)
