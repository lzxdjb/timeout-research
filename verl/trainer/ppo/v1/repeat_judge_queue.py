"""Opt-in repeat-only inference coordinator for V1 synchronous rollouts."""

import asyncio
import os
import time

import ray


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

    async def ready(self):
        from recipe.swe_agent import repetition_reward as repeat
        self.client = repeat.get_client(self.config)
        await self.client.preflight()

    async def reserve(self, job_id):
        started = time.monotonic()
        await self.capacity.acquire()
        if job_id in self.reservations:
            self.capacity.release()
            raise RuntimeError("Duplicate repeat job reservation: " + job_id)
        self.reservations.add(job_id)
        return time.monotonic() - started

    async def abandon(self, job_id):
        if job_id in self.running:
            raise RuntimeError("Cannot abandon a running repeat job")
        if job_id in self.reservations:
            self.reservations.remove(job_id)
            self.capacity.release()

    async def evaluate(self, job_id, job, admission_seconds=0.0):
        from recipe.swe_agent import repetition_reward as repeat

        if job_id not in self.reservations or job_id in self.running:
            raise RuntimeError("Missing or duplicate repeat reservation: " + job_id)
        self.running.add(job_id)
        started = time.monotonic()
        try:
            if job["config"] != self.config:
                raise RuntimeError("Repeat coordinator configuration mismatch")
            actor = job["actor_segments"]
            verdict = await self.client.judge_consensus(
                job["transcript"], job["events"], **({"actor_segments": actor} if actor is not None else {})
            )
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
        finally:
            self.running.remove(job_id)
            self.reservations.remove(job_id)
            self.capacity.release()


# No GPUs: inference remains on the configured external endpoint. No automatic replay of lost jobs.
RepeatJudgeCoordinatorActor = ray.remote(num_cpus=1, max_concurrency=10000, max_restarts=0)(RepeatJudgeCoordinator)
