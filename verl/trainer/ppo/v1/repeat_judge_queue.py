"""Opt-in repeat-only inference coordinator for V1 synchronous rollouts."""

import asyncio
import logging
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
    """One shared HTTP semaphore; reservations bound retained trajectory payloads.

    Capacity wait occurs at producers *before* sending large public trajectories.
    Ray actor failure is deliberately propagated rather than converted into a clean reward.
    """

    def __init__(self, config):
        self.config = config
        self.client = None
        self.capacity = asyncio.Semaphore(config.max_pending)
        self.reservations = set()
        self.running = set()
        self.waiting = set()
        self.debug = _debug_bool("SWE_AGENT_REPEAT_JUDGE_DEBUG")
        self.debug_slow_seconds = _debug_slow_seconds()
        self._last_debug_snapshot = 0.0
        if self.debug:
            logger.info(
                "Repeat judge debug startup endpoint=%s model=%s max_context=%s max_output=%s "
                "timeout=%s queue_timeout=%s concurrency=%s vote_count=%s vote_quorum=%s max_pending=%s",
                os.getenv("SWE_AGENT_REPEAT_JUDGE_URL", ""),
                os.getenv("SWE_AGENT_REPEAT_JUDGE_MODEL", ""),
                os.getenv("SWE_AGENT_REPEAT_JUDGE_MAX_CONTEXT", ""),
                os.getenv("SWE_AGENT_REPEAT_JUDGE_MAX_OUTPUT", ""),
                os.getenv("SWE_AGENT_REPEAT_JUDGE_TIMEOUT", ""),
                os.getenv("SWE_AGENT_REPEAT_JUDGE_QUEUE_TIMEOUT", ""),
                getattr(config, "concurrency", ""),
                getattr(config, "vote_count", ""),
                getattr(config, "vote_quorum", ""),
                getattr(config, "max_pending", ""),
            )

    def _debug_snapshot(self, reason: str, force: bool = False) -> None:
        if not self.debug:
            return
        now = time.monotonic()
        interval = max(1.0, min(self.debug_slow_seconds or 1.0, 30.0))
        if not force and now - self._last_debug_snapshot < interval:
            return
        self._last_debug_snapshot = now
        logger.info(
            "Repeat judge queue snapshot reason=%s reserved=%d running=%d waiting=%d capacity=%d",
            reason,
            len(self.reservations),
            len(self.running),
            len(self.waiting),
            getattr(self.capacity, "_value", "unknown"),
        )

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
            logger.info("Repeat judge debug preflight start endpoint=%s", os.getenv("SWE_AGENT_REPEAT_JUDGE_URL", ""))
        started = time.monotonic()
        await self.client.preflight()
        if self.debug:
            self._debug_duration("preflight", "startup", started)
            self._debug_snapshot("ready", force=True)

    async def reserve(self, job_id):
        started = time.monotonic()
        if self.debug:
            self.waiting.add(job_id)
            logger.info("Repeat judge reservation requested job_id=%s reserved=%d running=%d waiting=%d",
                        job_id, len(self.reservations), len(self.running), len(self.waiting))
        try:
            await self.capacity.acquire()
        finally:
            self.waiting.discard(job_id)
        if job_id in self.reservations:
            self.capacity.release()
            raise RuntimeError("Duplicate repeat job reservation: " + job_id)
        self.reservations.add(job_id)
        admission = time.monotonic() - started
        if self.debug:
            self._debug_duration("reserve_wait", job_id, started)
            self._debug_snapshot("reservation_acquired", force=admission >= self.debug_slow_seconds)
        return admission

    async def abandon(self, job_id):
        if job_id in self.running:
            raise RuntimeError("Cannot abandon a running repeat job")
        if job_id in self.reservations:
            self.reservations.remove(job_id)
            self.capacity.release()
        self._debug_snapshot("abandon", force=True)

    async def evaluate(self, job_id, job, admission_seconds=0.0):
        from recipe.swe_agent import repetition_reward as repeat

        if job_id not in self.reservations or job_id in self.running:
            raise RuntimeError("Missing or duplicate repeat reservation: " + job_id)
        self.running.add(job_id)
        started = time.monotonic()
        if self.debug:
            logger.info("Repeat judge evaluate start job_id=%s reserved=%d running=%d waiting=%d",
                        job_id, len(self.reservations), len(self.running), len(self.waiting))
        try:
            if job["config"] != self.config:
                raise RuntimeError("Repeat coordinator configuration mismatch")
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
            job = {**job, "metadata": dict(job["metadata"])}
            job["metadata"].update(
                repeat_reward_judge_called=1, repeat_reward_stage=2,
                repeat_reward_pending=0, repeat_reward_deferred=1,
            )
            score, metadata = await repeat.finish_repeat_reward(job, verdict)
            metadata.update(
                repeat_reward_deferred_admission_seconds=admission_seconds,
                repeat_reward_deferred_wall_seconds=time.monotonic() - started,
                repeat_reward_deferred_request_queue_seconds=verdict.get("queue_seconds", 0.0),
                repeat_reward_deferred_service_seconds=verdict.get("service_seconds", 0.0),
            )
            return job_id, score, metadata
        except Exception as error:
            if self.debug:
                logger.warning("Repeat judge evaluate error job_id=%s type=%s message=%s",
                               job_id, type(error).__name__, str(error)[:240])
            raise
        finally:
            self.running.remove(job_id)
            self.reservations.remove(job_id)
            self.capacity.release()
            if self.debug:
                self._debug_duration("evaluate_total", job_id, started)
                self._debug_snapshot("evaluation_released", force=True)


# No GPUs: inference remains on the configured external endpoint. No automatic replay of lost jobs.
RepeatJudgeCoordinatorActor = ray.remote(num_cpus=1, max_concurrency=10000, max_restarts=0)(RepeatJudgeCoordinator)
