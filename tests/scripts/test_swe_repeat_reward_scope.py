"""Network-free reward-scope contracts for valid successes and failures."""
import gzip
import json

import pytest

from recipe.swe_agent import repetition_reward as p


@pytest.mark.parametrize('scope', ['success_only', 'all_valid'])
@pytest.mark.parametrize('mode', ['off', 'shadow', 'apply'])
@pytest.mark.parametrize('validation', [False, True])
@pytest.mark.parametrize('valid', [False, True])
@pytest.mark.parametrize('verdict', ['clean', 'uncertain', 'repetitive'])
@pytest.mark.parametrize('raw,base,all_valid_penalty', [(1, 1, .1), (0, 0, -.1), (0, -.1, -.2), (1, -.1, -.2)])
def test_reward_table_and_default_compatibility(scope, mode, validation, valid, verdict, raw, base, all_valid_penalty):
    cfg = p.RepeatConfig(mode=mode, reward_scope=scope, validation_enabled=True)
    in_scope = scope == 'all_valid' or base == 1
    penalty = all_valid_penalty if valid and in_scope and verdict == 'repetitive' else base
    actual = p.compose_reward(cfg, validation=validation, valid=valid, raw=raw, existing=base, verdict=verdict)
    assert actual == pytest.approx(penalty if mode == 'apply' and not validation else base)
    assert p.hypothetical_reward(cfg, valid=valid, raw=raw, existing=base, verdict=verdict) == pytest.approx(penalty)
    assert p.eligible(cfg, validation=validation, valid=valid, raw=raw, existing=base) == (valid and in_scope and mode != 'off' and not validation)
    assert p.judgment_eligible(cfg, validation=validation, valid=valid, raw=raw, existing=base) == (valid and (validation or mode != 'off' and in_scope))


@pytest.mark.parametrize('penalty', [.05, .1, .3])
def test_additive_penalty_never_improves_nonpositive_reward(penalty):
    cfg = p.RepeatConfig(mode='apply', reward_scope='all_valid', nonpositive_penalty=penalty)
    for base in (0., -.1):
        assert p.compose_reward(cfg, validation=False, valid=True, raw=0, existing=base,
                                verdict='repetitive') == pytest.approx(base - penalty)


@pytest.mark.parametrize('name,value', [
    ('SWE_AGENT_REPEAT_REWARD_SCOPE', 'all'),
    ('SWE_AGENT_REPEAT_NONPOSITIVE_PENALTY', '0'),
    ('SWE_AGENT_REPEAT_NONPOSITIVE_PENALTY', '-0.1'),
    ('SWE_AGENT_REPEAT_NONPOSITIVE_PENALTY', 'nan'),
    ('SWE_AGENT_REPEAT_NONPOSITIVE_PENALTY', 'inf'),
])
def test_invalid_scope_configuration(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        p.RepeatConfig.from_env()


def test_configuration_defaults_and_worker_forwarding(monkeypatch):
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    monkeypatch.delenv('SWE_AGENT_REPEAT_REWARD_SCOPE', raising=False)
    monkeypatch.delenv('SWE_AGENT_REPEAT_NONPOSITIVE_PENALTY', raising=False)
    assert p.RepeatConfig.from_env().reward_scope == 'success_only'
    monkeypatch.setenv('SWE_AGENT_REPEAT_REWARD_SCOPE', 'all_valid')
    monkeypatch.setenv('SWE_AGENT_REPEAT_NONPOSITIVE_PENALTY', '0.05')
    cfg = p.RepeatConfig.from_env()
    assert cfg.reward_scope == 'all_valid' and cfg.nonpositive_penalty == .05
    forwarded = get_ppo_ray_runtime_env()['env_vars']
    assert forwarded['SWE_AGENT_REPEAT_REWARD_SCOPE'] == 'all_valid'
    assert forwarded['SWE_AGENT_REPEAT_NONPOSITIVE_PENALTY'] == '0.05'


def test_scope_is_auditable_but_never_sent_to_judge(monkeypatch, tmp_path):
    monkeypatch.setenv('SWE_AGENT_REPEAT_AUDIT_DIR', str(tmp_path))
    cfg = p.RepeatConfig(mode='shadow', reward_scope='all_valid', nonpositive_penalty=.05)
    artifact = p.save_audit('', [], p.uncertain('fixture'), cfg)
    with gzip.open(artifact['repeat_reward_audit_path'], 'rt') as stream:
        saved = json.load(stream)
    assert saved['config']['reward_scope'] == 'all_valid'
    assert saved['config']['nonpositive_penalty'] == .05
    provenance = p.JudgeClient(cfg).provenance()
    assert provenance['reward_scope'] == 'all_valid'
    assert provenance['nonpositive_penalty'] == .05
    assert p.completion_request(cfg, '', [], seed=0) == p.completion_request(p.RepeatConfig(), '', [], seed=0)


@pytest.mark.parametrize('raw,base', [(float('nan'), 0), (0, float('inf')), (float('inf'), -.1)])
def test_nonfinite_rewards_are_never_eligible(raw, base):
    cfg = p.RepeatConfig(mode='apply', reward_scope='all_valid')
    assert not p.eligible(cfg, validation=False, valid=True, raw=raw, existing=base)


def test_offline_replay_reports_failed_and_nonsubmitted_coverage(tmp_path, monkeypatch):
    import asyncio
    from scripts.swe_repeat_regression import evaluate_two_level
    monkeypatch.setenv('SWE_AGENT_REPEAT_REWARD_SCOPE', 'all_valid')
    monkeypatch.setenv('SWE_AGENT_REPEAT_NONPOSITIVE_PENALTY', '0.1')
    monkeypatch.setenv('SWE_AGENT_REPEAT_AUDIT_DIR', str(tmp_path / 'audits'))
    args = {'file_path': 'a.py', 'old_string': 'same', 'new_string': 'same'}
    events = [{'id': i, 'name': 'Edit', 'arguments': args, 'completed': True, 'serial': True,
               **p.observation_fields(p.NOOP_EDIT_RESULT),
               **p.capture_observation(p.NOOP_EDIT_RESULT),
               **p.noop_edit_evidence('Edit', args, p.NOOP_EDIT_RESULT)} for i in range(1, 6)]
    fixture = tmp_path / 'fixtures.jsonl.gz'
    with gzip.open(fixture, 'wt') as stream:
        for submitted, base in [(True, 0.), (False, -.1)]:
            stream.write(json.dumps({'id': f'synthetic:submitted:{submitted}', 'task_key': f'task:{submitted}',
                'partition': 'development', 'public_events': events, 'public_context': 'Public repair task',
                'raw': {'raw_score': 0., 'score': base, 'submission_signal_seen': submitted}}) + '\n')
    labels = tmp_path / 'labels.json'
    labels.write_text(json.dumps({'policy_version': p.POLICY_VERSION, 'cases': [
        {'id': f'synthetic:submitted:{s}', 'expected': 'repetitive',
         'reason': 'Five serial explicitly rejected identical no-op edits.'} for s in (True, False)]}))
    report = asyncio.run(evaluate_two_level(fixture, labels, tmp_path / 'report.json'))
    assert report['by_raw_outcome']['failure']['tp'] == 2
    assert report['by_raw_outcome']['success'].get('total', 0) == 0
    assert report['provenance']['reward_scope'] == 'all_valid'
    assert report['stage1_rule_matches']['explicit_noop_edits'] == 2
