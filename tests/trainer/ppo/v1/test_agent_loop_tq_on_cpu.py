# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import asyncio

from verl.trainer.ppo.v1.agent_loop_tq import _infrastructure_failure_from_extra_fields, _settle_session_tasks


def test_settle_session_tasks_waits_for_siblings_after_failure():
    async def run():
        settled = asyncio.Event()

        async def fail():
            raise RuntimeError("session failed")

        async def finish_later():
            await asyncio.sleep(0.01)
            settled.set()

        tasks = [asyncio.create_task(fail()), asyncio.create_task(finish_later())]
        errors = await _settle_session_tasks(tasks)

        assert settled.is_set()
        assert all(task.done() for task in tasks)
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)

    asyncio.run(run())


def test_settle_session_tasks_waits_for_cancelled_sessions():
    async def run():
        cleanup_complete = asyncio.Event()

        async def session():
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                cleanup_complete.set()

        task = asyncio.create_task(session())
        await asyncio.sleep(0)
        task.cancel()
        errors = await _settle_session_tasks([task])

        assert cleanup_complete.is_set()
        assert len(errors) == 1
        assert isinstance(errors[0], asyncio.CancelledError)

    asyncio.run(run())


def test_infrastructure_failure_marker_supports_nested_and_direct_metadata():
    assert _infrastructure_failure_from_extra_fields(
        {"reward_extra_info": {"observed_infrastructure_failure": 1}}
    )
    assert _infrastructure_failure_from_extra_fields({"infrastructure_failure": 1})
    assert not _infrastructure_failure_from_extra_fields({"observed_infrastructure_failure": 0})
