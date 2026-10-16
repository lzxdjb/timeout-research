"""Boundary rendering and historical request integrity; no model calls."""
import copy
import gzip
import json
from pathlib import Path

import pytest

from recipe.swe_agent import repetition_reward as p
from recipe.swe_agent import repeat_legacy_v13 as old
from scripts.swe_repeat_regression import replay_evidence


@pytest.mark.parametrize('boundary', [
    '</think>\n<|im_end|>\n', '<think>\r\n<|endoftext|>\r\n',
    '</think><|im_end|><|endoftext|>\n',
])
def test_internal_boundary_runs_are_removed(boundary):
    assert not p.actor_judge_text(boundary * 100).strip()
    assert p.actor_judge_text('First thought.\n' + boundary * 100 + 'Next thought.') == 'First thought.\nNext thought.'


@pytest.mark.parametrize('text', [
    '```\n</think>\n<|im_end|>',  # unclosed source fence
    '````text\n```\n</think>\n<|im_end|>\n````',
    '~~~text\n```\n</think>\n<|endoftext|>\n~~~',
    '> </think>\n> <|im_end|>\n',
    '    </think>\n    <|im_end|>\n',
    '\t</think>\n\t<|endoftext|>\n',
    'The literal "</think>" and `<|im_end|>` are fixture data.',
    'quoted suffix <|im_end|>',
])
def test_literal_examples_survive_exactly(text):
    assert p.actor_judge_text(text) == text


def test_nested_fence_then_real_boundaries_and_wrapped_thinking():
    literal = '````text\n```\n</think>\n<|im_end|>\n````\n'
    raw = '<think>Inspect the fixture.\n' + literal + '</think>\n<|im_end|>\nContinue.'
    assert p.actor_judge_text(raw) == 'Inspect the fixture.\n' + literal + 'Continue.'
    assert p.actor_judge_text('<think>The literal "</think>" stays.</think>Final.') == 'The literal "</think>" stays.Final.'
    assert p.actor_judge_text('<think>Inspect.</think><|im_end|>') == 'Inspect.'
    assert p.actor_judge_text('> literal</think>\n<|im_end|>') == '> literal</think>\n'


def test_rendering_keeps_raw_audit_and_guard_contract(tmp_path, monkeypatch):
    raw = '</think>\n<|im_end|>\n' * 100
    segments = [{'id': 'a1', 'turn': 1, 'after_event_id': 0, 'text': raw,
                 'captured': True, 'source': 'generated_tokens'}]
    untouched = copy.deepcopy(segments)
    value = {'verdict': 'repetitive', 'category': 'repetitive_reasoning',
             'event_ids': [], 'text_ids': ['a1'], 'reason': 'Existing availability contract.'}
    assert p.validate_verdict(value, [], segments) == value
    assert p.evidence_guard(value, [], segments) == value  # guard intentionally unchanged
    full = json.loads(p.judge_messages('', [], actor_segments=segments)[1]['content'])
    compact = json.loads(p.judge_messages('', [], actor_segments=segments,
                                         encoding=p.COMPACT_ENCODING_VERSION)[1]['content'])
    assert p.decode_compact_body(compact) == full
    assert full['trajectory'][0]['text'] == ''
    monkeypatch.setenv('SWE_AGENT_REPEAT_AUDIT_DIR', str(tmp_path))
    audit = p.save_audit('', [], value, p.RepeatConfig(), actor_segments=segments)
    with gzip.open(audit['repeat_reward_audit_path'], 'rt') as stream:
        saved = json.load(stream)
    assert saved['actor_segments'][0]['text'] == raw
    assert segments == untouched


def test_saved_v13_request_replays_with_frozen_serializer():
    report = Path('analysis/reward_vote_sweep/v13_thinking_ablation_20261016T031250_3466802/baseline/report.json')
    if not report.exists():
        pytest.skip('Local historical audit is unavailable')
    rows = json.loads(report.read_text())['rows']
    row = next(x for x in rows if x['id'] == 'synthetic:trajectory-v13:boundary_only')
    events, transcript, source, segments = replay_evidence(row, p)
    assert source == 'captured_v13_migrated'
    assert old.SERIALIZER_VERSION == 'repeat-public-trajectory-v5'
    assert old.actor_judge_text(segments[0]['text']).count('<|im_end|>') == 99
    assert p.actor_judge_text(segments[0]['text']) == ''
