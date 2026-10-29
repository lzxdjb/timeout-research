# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Teacher-forced scoring through an already deployed vLLM HTTP server.

Only sampled-token distillation is supported. The student transcript is sent as
token IDs without chat-template rendering, tokenization, or special-token insertion.
The caller must use compatible text tokenizers and must not send audio trajectories.
"""

import asyncio
import logging
import math
import os
from typing import Any, Optional

import aiohttp

from verl.experimental.teacher_loop.errors import TeacherScoringError
from verl.workers.config.distillation import ExternalTeacherConfig
from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__name__)


def parse_prompt_logprobs(payload: dict[str, Any], prompt_ids: list[int]) -> dict[str, list]:
    """Validate exact token alignment and retain the supplied token's score.

    vLLM's prompt_logprobs=0 still includes the actual input token, even when
    it is outside the most likely tokens. The first position has no causal
    predecessor. Match the managed teacher's next-token convention: discard
    that position, shift scores/IDs left, and append an unused finite dummy.
    """
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("External teacher must return exactly one completion choice.")
    choice = choices[0]
    if choice.get("prompt_token_ids") != prompt_ids:
        raise ValueError("External teacher changed prompt token IDs or omitted prompt_token_ids.")
    scores = choice.get("prompt_logprobs")
    if not isinstance(scores, list) or len(scores) != len(prompt_ids):
        raise ValueError("External teacher returned a different number of prompt log probabilities.")
    logprobs = []
    for position, (token_id, token_scores) in enumerate(zip(prompt_ids, scores, strict=True)):
        if position == 0 and token_scores is None:
            logprobs.append([0.0])
            continue
        if not isinstance(token_scores, dict) or str(token_id) not in token_scores:
            raise ValueError(f"External teacher omitted the supplied token at position {position}.")
        logprob = token_scores[str(token_id)].get("logprob")
        if isinstance(logprob, bool) or not isinstance(logprob, (float, int)):
            raise ValueError(f"External teacher returned an invalid log probability at position {position}.")
        if not math.isfinite(logprob) or logprob > 1e-5:
            raise ValueError(f"External teacher returned a nonfinite or positive log probability at {position}.")
        logprobs.append([float(logprob)])
    return {
        "prompt_ids": [[token_id] for token_id in prompt_ids[1:]] + [[0]],
        "prompt_logprobs": logprobs[1:] + [[0.0]],
    }


class ExternalTeacherScorer:
    """One shared async Ray actor per teacher, with bounded global concurrency.

    Kept as a plain class so HTTP alignment, failures, and concurrency can be
    exercised without starting Ray. The manager wraps it with ``ray.remote``.
    """

    def __init__(self, config: ExternalTeacherConfig):
        config.validate()
        self.config = config
        self._base_url = config.base_url.rstrip("/")
        if not self._base_url.endswith("/v1"):
            self._base_url += "/v1"
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self._session: Optional[aiohttp.ClientSession] = None
        self._unsupported_ids = set(config.unsupported_token_ids)

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {}
            if self.config.api_key_env:
                api_key = os.environ.get(self.config.api_key_env)
                if not api_key:
                    raise ValueError(
                        f"Missing external teacher API key environment variable {self.config.api_key_env}."
                    )
                headers["Authorization"] = f"Bearer {api_key}"
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout_s),
                connector=aiohttp.TCPConnector(limit=self.config.max_concurrent_requests),
            )
        return self._session

    async def _request(self, method: str, path: str, payload: Optional[dict] = None) -> dict:
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self._get_session().request(method, self._base_url + path, json=payload) as response:
                    if response.status == 429 or response.status == 408 or response.status >= 500:
                        # Retriable service errors are never replaced with fake teacher targets.
                        response.raise_for_status()
                    if response.status != 200:
                        raise RuntimeError(f"External teacher rejected {path} with HTTP {response.status}.")
                    result = await response.json()
                    if not isinstance(result, dict):
                        raise ValueError("External teacher returned a non-object JSON response.")
                    return result
            except (aiohttp.ClientConnectionError, aiohttp.ClientResponseError, asyncio.TimeoutError) as exc:
                if attempt == self.config.max_retries:
                    raise RuntimeError(
                        f"External teacher {path} failed after {attempt + 1} attempt(s); no targets produced."
                    ) from exc
                logger.warning(
                    "External teacher request failed; retrying (%d/%d).", attempt + 1, self.config.max_retries
                )
                await asyncio.sleep(self.config.retry_backoff_s * (2**attempt))

    async def validate_connection(self) -> dict:
        """Fail early for an incorrect served-model alias or context limit."""
        async with self._semaphore:
            payload = await self._request("GET", "/models")
        model = next((model for model in payload.get("data", []) if model.get("id") == self.config.model_name), None)
        if model is None:
            raise ValueError(f"External teacher does not serve model {self.config.model_name!r}.")
        served_limit = model.get("max_model_len")
        if served_limit is not None and self.config.max_model_len > served_limit:
            raise ValueError(
                f"Configured external teacher context {self.config.max_model_len} exceeds server limit {served_limit}. "
                "Prompt-logprob scoring reserves one generated token. Increase the actual teacher serving limit "
                "before training; changing only the client configuration cannot increase server capacity."
            )
        return {"model_name": self.config.model_name, "max_model_len": served_limit}

    async def score(self, prompt_ids: list[int], sampling_params: dict[str, Any]) -> dict[str, list]:
        if not prompt_ids or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in prompt_ids
        ):
            raise ValueError("External teacher requires a nonempty list of nonnegative token IDs.")
        if len(prompt_ids) + 1 > self.config.max_model_len:
            raise ValueError("External teacher context has no room for the prompt and one scoring token.")
        if self._unsupported_ids.intersection(prompt_ids):
            raise ValueError("Transcript contains token IDs unsupported by the external teacher tokenizer.")
        if sampling_params.get("prompt_logprobs", 0) != 0:
            raise NotImplementedError("External teacher scoring currently supports sampled-token logprobs only.")
        payload = {
            "model": self.config.model_name,
            "prompt": [prompt_ids],
            "prompt_logprobs": 0,
            "max_tokens": 1,
            "temperature": 1.0,
            "add_special_tokens": False,
            "return_token_ids": True,
        }
        async with self._semaphore:
            result = await self._request("POST", "/completions", payload)
        return parse_prompt_logprobs(result, prompt_ids)

    async def close(self):
        if self._session is not None:
            await self._session.close()


class ExternalTeacherClient(LLMServerClient):
    """Serializable rollout-worker proxy to the shared HTTP scoring actor."""

    def __init__(self, config, scoring_handle):
        super().__init__(config=config)
        self._scoring_handle = scoring_handle

    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data=None,
        video_data=None,
        audio_data=None,
        mm_processor_kwargs=None,
        **kwargs,
    ) -> TokenOutput:
        try:
            if any(data is not None and len(data) > 0 for data in (image_data, video_data, audio_data)):
                raise NotImplementedError("External teacher OPD currently supports text-only trajectories.")
            extra_fields = await self._scoring_handle.score.remote(prompt_ids, sampling_params)
            return TokenOutput(token_ids=[], extra_fields=extra_fields)
        except Exception as exc:
            # V1 normally treats agent exceptions as failed rollouts and may
            # refill them. Teacher outages and malformed targets must instead
            # use its fatal postprocessing-error channel. Cancellation remains
            # cancellation (CancelledError derives from BaseException).
            raise TeacherScoringError(f"External teacher scoring failed: {str(exc)[:1000]}") from exc
