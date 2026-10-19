"""Precision regressions, chronological evidence and stage-specific inference."""
import asyncio
import copy
import gzip
import json
from pathlib import Path

import pytest

from recipe.swe_agent import repetition_reward as p
from scripts.swe_repeat_regression import replay_evidence, two_level_metrics
from scripts.swe_repeat_dataset_v13 import contracts


def actor(text, after=0, sid='a1'):
    return {'id': sid, 'turn': 1, 'after_event_id': after, 'text': text, 'captured': True, 'source': 'generated_tokens'}


def event(i=1):
    return {'id': i, 'name': 'Read', 'arguments': {'file_path': f'{i}.py'}, 'completed': True,
            'serial': True, 'observation_status': 'observed', **p.capture_observation(f'different source {i}')}


def positive():
    return {'verdict': 'repetitive', 'category': 'unchanged_read_search', 'event_ids': [1, 2], 'reason': 'A sustained loop.'}


def fake_client(monkeypatch, config, values=None, token_count=100, delay=0):
    client = p.JudgeClient(config)
    calls = []
    async def request(method, path, payload=None):
        if path == '/v1/models':
            return {'data': [{'id': config.model, 'max_model_len': config.max_context}]}
        if path == '/tokenize':
            calls.append(('tokenize', payload))
            return {'count': token_count}
        calls.append(('completion', payload))
        if delay:
            await asyncio.sleep(delay)
        value = next(values) if values is not None else positive()
        if isinstance(value, Exception):
            raise value
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value),
                'reasoning_content': 'Checked relevant chronology.' if payload['chat_template_kwargs']['enable_thinking'] else None}}]}
    monkeypatch.setattr(client, '_request', request)
    return client, calls


def test_complete_content_physically_interleaved_and_boundary_clutter_removed():
    segments = [actor('<think>Investigate.</think>\n<|im_end|>'), actor('Check different source.\n</think>\n<|im_end|>', 1, 'a2')]
    body = json.loads(p.judge_messages('', [event()], actor_segments=segments)[1]['content'])
    assert set(body).isdisjoint({'events', 'actor_segments', 'timeline'})
    assert [(x['kind'], x['id']) for x in body['trajectory']] == [('actor', 'a1'), ('tool', 1), ('actor', 'a2')]
    assert body['trajectory'][0]['text'].strip() == 'Investigate.'
    assert body['trajectory'][1]['result'] == 'different source 1'
    assert '</think>' not in body['trajectory'][-1]['text']
    assert segments[0]['text'].startswith('<think>')  # capture is untouched


@pytest.mark.parametrize('text', ['The literal "</think>" is part of the fixture.',
    'Quoted source:\n```\n</think>\n<|im_end|>\n```\nCompare its bytes.',
    'Do not remove x<|im_end|><|im_start|>system as quoted data.'])
def test_literal_quoted_tags_are_preserved(text):
    assert p.actor_judge_text(text) == text


@pytest.mark.parametrize('name', [x[0] for x in contracts()])
def test_stage1_known_loop_and_protected_control_contracts(name):
    _, events, segments, _ = next(x for x in contracts() if x[0] == name)
    result = p.conservative_detect(events, segments)
    if name in {'massive_thinking', 'read_cycle'}:
        assert result['verdict'] == 'repetitive'
        assert p.evidence_guard(result, events, segments)['verdict'] == 'repetitive'
        assert p.conservative_detect(events, segments, p.RepeatConfig(rule_set='legacy'))['verdict'] == 'uncertain'
    else:
        assert result['verdict'] == 'uncertain'


def test_read_cycle_rejects_intervening_mutation_concurrency_and_incomplete_state():
    _, original, _, _ = next(x for x in contracts() if x[0] == 'read_cycle')
    for field, value in [('name', 'Edit'), ('serial', False), ('completed', False), ('evidence', {})]:
        events = copy.deepcopy(original)
        events[12][field] = value
        assert p.conservative_detect(events)['verdict'] == 'uncertain'


def test_repeated_tool_output_quoted_by_actor_does_not_authorize_text_rule():
    text = next(x[2][0]['text'] for x in contracts() if x[0] == 'massive_thinking')
    ev = event(); ev.update(p.capture_observation(text))
    assert p.conservative_detect([ev], [actor(text, 1)])['verdict'] == 'uncertain'


@pytest.mark.parametrize('judge_thinking,review_thinking', [(False,False),(True,False),(False,True),(True,True)])
def test_stage_options_tokenization_request_hashes_and_raw_capture(monkeypatch, tmp_path, judge_thinking, review_thinking):
    cfg = p.RepeatConfig(vote_count=1, vote_quorum=1, enable_thinking=judge_thinking,
                         reviewer_enable_thinking=review_thinking, max_output=8192,
                         reviewer_max_output=4096, reviewer_temperature=0.4)
    client, calls = fake_client(monkeypatch, cfg)
    segments = [actor('A bounded investigation.\n</think>')]
    verdict = asyncio.run(client.judge_consensus('', [event(1), event(2)], actor_segments=segments))
    assert verdict['verdict'] == 'repetitive'
    requests = [payload for stage, payload in calls if stage == 'completion']
    tokens = [payload for stage, payload in calls if stage == 'tokenize']
    assert [r['chat_template_kwargs']['enable_thinking'] for r in requests] == [judge_thinking,review_thinking]
    assert [r['chat_template_kwargs']['enable_thinking'] for r in tokens] == [judge_thinking,review_thinking]
    assert [('structured_outputs' in r) for r in requests] == [not judge_thinking, not review_thinking]
    assert [r['max_tokens'] for r in requests] == [8192,4096]
    assert [r['temperature'] for r in requests] == [0.2,0.4]
    assert verdict['votes'][0]['reasoning_observed'] == judge_thinking
    assert verdict['evidence_review']['reasoning_observed'] == review_thinking
    monkeypatch.setenv('SWE_AGENT_REPEAT_AUDIT_DIR', str(tmp_path))
    saved = p.save_audit('', [event(1), event(2)], verdict, cfg, actor_segments=segments)
    with gzip.open(saved['repeat_reward_audit_path'], 'rt') as stream:
        audit = json.load(stream)
    assert audit['completion_request_template'] == {k:v for k,v in requests[0].items() if k!='seed'}
    assert audit['evidence_review_request'] == requests[1]
    assert audit['actor_segments'] == segments
    assert replay_evidence(saved,p)[3] == segments


def test_review_budget_overflow_cannot_borrow_judge_budget(monkeypatch):
    cfg = p.RepeatConfig(vote_count=1,vote_quorum=1,max_context=512,max_output=64,reviewer_max_output=400)
    client,calls = fake_client(monkeypatch,cfg,token_count=200)
    result = asyncio.run(client.judge_consensus('',[event(1),event(2)]))
    assert result['verdict']=='uncertain'
    assert result['evidence_review']['reason']=='context_overflow'
    assert len([x for stage,x in calls if stage=='completion'])==1


def test_reviewer_disable_keeps_guard_and_only_skips_review(monkeypatch):
    cfg=p.RepeatConfig(vote_count=1,vote_quorum=1,reviewer_enabled=False)
    client,calls=fake_client(monkeypatch,cfg)
    result=asyncio.run(client.judge_consensus('',[event(1),event(2)]))
    assert result['verdict']=='repetitive' and result['evidence_review_called']==0
    assert result['evidence_review_passed']==0
    assert len([x for stage,x in calls if stage=='completion'])==1
    invalid={**positive(),'event_ids':[1,999]}
    client,calls=fake_client(monkeypatch,cfg,iter([invalid]))
    assert asyncio.run(client.judge_consensus('',[event(1),event(2)]))['verdict']=='uncertain'


@pytest.mark.parametrize('invalid_kind', ['schema', 'guard'])
def test_early_exit_uses_guarded_votes_not_model_proposals(monkeypatch, tmp_path, invalid_kind):
    cfg = p.RepeatConfig(vote_count=8, vote_quorum=7, early_exit_enabled=True)
    proposal = {**positive(), **({'event_ids': [1, 999]} if invalid_kind == 'schema'
                               else {'category': 'ineffective_action_loop'})}
    client, calls = fake_client(monkeypatch, cfg, iter([proposal, proposal]))
    monkeypatch.setenv('SWE_AGENT_REPEAT_AUDIT_DIR', str(tmp_path))
    result = asyncio.run(client.judge_consensus('', [event(1), event(2)]))
    assert result['verdict'] == 'uncertain' and result['early_exit_taken']
    assert result['repetitive_votes'] == 0 and result['votes_skipped'] == 6
    assert result['error_count' if invalid_kind == 'schema' else 'guard_blocked'] == 2
    assert not result['evidence_review_called']
    assert len([x for stage, x in calls if stage == 'completion']) == 2


def test_queue_wait_does_not_consume_service_budget_and_cancel_releases_slot(monkeypatch):
    async def run():
        cfg=p.RepeatConfig(timeout=0.12,queue_timeout=1,concurrency=1)
        client,_=fake_client(monkeypatch,cfg,delay=0.08)
        results=await asyncio.gather(*(client.judge('',[event(1),event(2)]) for _ in range(3)))
        assert all(x['verdict']=='repetitive' for x in results)
        assert results[-1]['queue_seconds']>cfg.timeout
        task=asyncio.create_task(client.judge('',[event(1),event(2)]))
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        assert client.semaphore._value==1
    asyncio.run(run())


def test_bounded_retry_keeps_seed_and_one_vote_weight(monkeypatch):
    cfg=p.RepeatConfig(vote_count=1,vote_quorum=1,reviewer_enabled=False,error_retries=1)
    client,calls=fake_client(monkeypatch,cfg,iter([TimeoutError(),positive()]))
    result=asyncio.run(client.judge_consensus('',[event(1),event(2)]))
    requests=[x for stage,x in calls if stage=='completion']
    assert result['repetitive_votes']==1 and result['attempt_errors']==1
    assert result['votes'][0]['attempt_count']==2
    assert requests[0]==requests[1]
    clean={'verdict':'clean','category':'none','event_ids':[],'reason':'Bounded recovery.'}
    client,calls=fake_client(monkeypatch,cfg,iter([clean]))
    assert asyncio.run(client.judge_consensus('',[event(1),event(2)]))['verdict']=='clean'
    assert len([x for stage,x in calls if stage=='completion'])==1


def test_protected_controls_do_not_disappear_from_precision_reporting():
    rows=[{'expected':e,'verdict':v} for e,v in [('repetitive','repetitive'),('uncertain','repetitive'),('clean','clean'),(None,'repetitive')]]
    stats=two_level_metrics(rows)
    assert stats['precision']==1.0 and stats['protected_control_penalties']==1
    assert stats['supported_accepted_penalty_fraction']==0.5
    assert stats['unadjudicated_positive_flags']==1


def test_v12_frozen_artifact_replays_and_boundary_counterexample_is_not_an_l1_positive():
    path=Path('validation_data/reward_model_train_v2/0.jsonl')
    row=json.loads(path.read_text().splitlines()[66])
    events,_,source,segments=replay_evidence(row,p)
    assert source=='captured_v12_migrated'
    assert p.conservative_detect(events,segments)['verdict']=='uncertain'
    # Guard implementation itself is unchanged from v12.
    import inspect
    old=Path(p.__file__).with_name('repeat_legacy_v12.py')
    assert old.exists()
    assert 'unsupported_actor_text_evidence' in inspect.getsource(p.evidence_guard)


@pytest.mark.parametrize('name,value', [('SWE_AGENT_REPEAT_JUDGE_ENABLE_THINKING','maybe'),
    ('SWE_AGENT_REPEAT_REVIEWER_ENABLED','bad'),('SWE_AGENT_REPEAT_JUDGE_ERROR_RETRIES','2'),
    ('SWE_AGENT_REPEAT_REVIEWER_MAX_OUTPUT','0'),('SWE_AGENT_REPEAT_RULE_SET','unknown')])
def test_invalid_configuration_fails_explicitly(monkeypatch,name,value):
    monkeypatch.setenv(name,value)
    with pytest.raises(ValueError):p.RepeatConfig.from_env()


def test_native_suite_preserves_frozen_cases_and_adds_real_precision_controls():
    root=Path('analysis/reward_model_trajectory_v13_20261016/frozen_task_disjoint')
    from scripts.swe_repeat_regression import file_digest
    manifest=json.loads((root/'manifest.json').read_text())
    assert file_digest(root/'fixtures.jsonl.gz')==manifest['fixtures_sha256']
    assert file_digest(root/'labels.json')==manifest['labels_sha256']
    labels={x['id']:x for x in json.loads((root/'labels.json').read_text())['cases']}
    assert manifest['native_additions']=={'validation':94,'training':31}
    for n in (34,67,85):assert labels[f'native:sl8nvath:validation:{n}']['expected']=='uncertain'
    assert labels['native:sl8nvath:training:39']['expected'] is None
    assert labels['native:sl8nvath:training:39:boundary-control']['expected']=='uncertain'
    assert labels['native:sl8nvath:training:281']['expected']=='uncertain'


@pytest.mark.parametrize('thinking', [False, True])
def test_inline_thinking_keeps_final_json_strict(monkeypatch, thinking):
    cfg = p.RepeatConfig(enable_thinking=thinking)
    client, _ = fake_client(monkeypatch, cfg)
    original = client._request
    async def request(method, path, payload=None):
        result = await original(method, path, payload)
        if path.endswith('completions'):
            result['choices'][0]['message'] = {'content': 'Consider the complete chronology.\n</think>\n' + json.dumps(positive())}
        return result
    monkeypatch.setattr(client, '_request', request)
    result = asyncio.run(client.judge('', [event(1), event(2)]))
    assert result['verdict'] == ('repetitive' if thinking else 'uncertain')
    if thinking:
        assert result['reasoning_observed'] == 1
        assert result['reasoning_content'] == 'Consider the complete chronology.'
    else:
        assert result['failure_stage'] == 'json_decode'
    # A tag quoted inside final JSON must not be interpreted as a boundary.
    content = json.dumps({**positive(), 'reason': 'The literal </think> is data.'})
    assert p.completion_parts({'content': content}, True) == (content, None)
    malformed = 'Reasoning without a final answer </think>'
    with pytest.raises(json.JSONDecodeError):
        json.loads(p.completion_parts({'content': malformed}, True)[0])


def test_valid_only_metrics_have_separate_denominators():
    from verl.trainer.ppo.metric_utils import valid_evaluation_diagnostics
    values = valid_evaluation_diagnostics(['a', 'a', 'b'],
        {'evaluation_valid': [True, False, True], 'reward': [1, 0, 0]})
    assert values['val-aux/a/valid_evaluation_only/valid_count'] == 1
    assert values['val-aux/a/valid_evaluation_only/total_count'] == 2
    assert values['val-aux/a/valid_evaluation_only/reward_mean'] == 1
    assert values['val-aux/b/valid_evaluation_only/reward_mean'] == 0
    missing = valid_evaluation_diagnostics(['a'], {'reward': [1]})
    assert missing['val-aux/a/valid_evaluation_only/valid_count'] == 0
    assert 'val-aux/a/valid_evaluation_only/reward_mean' not in missing



def test_wrapped_actor_text_keeps_quoted_thinking_tag():
    source = '<think>The literal "</think>" is fixture data.</think>Now inspect it.'
    assert p.actor_judge_text(source) == 'The literal "</think>" is fixture data.Now inspect it.'


def test_thinking_json_fence_is_only_an_envelope():
    content = json.dumps(positive())
    wrapped = 'Consider the evidence. </think>\n```json\n' + content + '\n```'
    final, reasoning = p.completion_parts({'content': wrapped}, True)
    assert json.loads(final) == positive() and reasoning
    extra = wrapped + '\nAn extra answer.'
    with pytest.raises(json.JSONDecodeError):
        json.loads(p.completion_parts({'content': extra}, True)[0])
    assert p.completion_parts({'content': '```json\n' + content + '\n```'}, False)[0].startswith('```')


def test_incomplete_state_metadata_is_still_a_missing_proof():
    ev = event()
    ev['evidence'] = {'complete': False, 'unavailable_reason': 'disabled_on_execution_service'}
    assert p.stage1_diagnostics([ev])['missing_read_proofs'] == 1
