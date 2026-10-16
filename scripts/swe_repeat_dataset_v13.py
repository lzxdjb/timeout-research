"""Freeze native v12 captures and audited controls for v13; no model requests."""
from __future__ import annotations

import argparse
import collections
import copy
import gzip
import json
from pathlib import Path

from scripts.swe_repeat_regression import digest, file_digest, partition, replay_evidence, write_json


def contracts():
    thought = "I should inspect the same place again because my current hypothesis has not changed and no new information is available."
    loop = (thought + "\n\n") * 100
    for name, text, expected in [
        ("massive_thinking", loop, "repetitive"),
        ("quoted_loop", "I am checking fixture data.\n```\n" + loop + "```\nThis is deliberately repeated input.", "clean"),
        ("quoted_prose", 'The quoted fixture is:\n' + ('> ' + thought + '\n\n') * 100, "clean"),
        ("bounded_thinking", (thought + "\n\n") * 3 + "A different observation resolves the hypothesis.", "uncertain"),
        ("boundary_only", ("</think>\n<|im_end|>\n") * 100, "uncertain"),
        ("useful_enumeration", '\n\n'.join(f'Boundary case {i} changes the expected index to {i + 1} and requires its own distinct assertion.' for i in range(200)), "clean"),
    ]:
        yield name, [], [{"id": "a1", "after_event_id": 0, "turn": 1, "text": text,
                          "captured": True, "source": "synthetic"}], expected
    # Exact static read-cycle, plus negative controls for state changes and missing proof.
    from recipe.swe_agent import repetition_reward as p
    events = []
    for i in range(24):
        result = f'source for file {i % 3}'
        args = {"file_path": f'file{i % 3}.py'}
        events.append({"id": i + 1, "name": "Read", "arguments": args, "completed": True,
                       "serial": True, "observation_status": "observed", **p.capture_observation(result),
                       "evidence": {"version": p.EVIDENCE_VERSION, "complete": True,
                                    "before": f'state{i % 3}', "after": f'state{i % 3}',
                                    "arguments_sha256": digest(args),
                                    "observation_sha256": __import__('hashlib').sha256(result.encode()).hexdigest()}})
    yield 'read_cycle', events, None, 'repetitive'
    missing = [{k: v for k, v in e.items() if k != 'evidence'} for e in events]
    # Missing state is not itself a clean label; it cannot authorize an automatic rule.
    yield 'read_cycle_missing_state', missing, None, None
    changed = copy.deepcopy(events)
    for i, event in enumerate(changed):
        event['evidence']['before'] = event['evidence']['after'] = f'changing-state-{i}'
        event['result'] = event['observation'] = f'new diagnostic information {i}'
        event['evidence']['observation_sha256'] = __import__('hashlib').sha256(event['result'].encode()).hexdigest()
    yield 'read_cycle_changing_state', changed, None, 'clean'


def build(base: Path, validation: Path, training: Path, output: Path):
    from recipe.swe_agent import repetition_reward as p
    if output.exists():
        raise ValueError('Refusing to overwrite a frozen suite')
    manifest = json.loads((base / 'manifest.json').read_text())
    if any(file_digest(base / name) != manifest[key] for name, key in
           [('fixtures.jsonl.gz', 'fixtures_sha256'), ('labels.json', 'labels_sha256')]):
        raise ValueError('Base suite integrity mismatch')
    with gzip.open(base / 'fixtures.jsonl.gz', 'rt') as stream:
        cases = list(map(json.loads, stream))
    labels = json.loads((base / 'labels.json').read_text())['cases']
    partitions = {c['task_key']: c['partition'] for c in cases}
    native_positives = {1, 2, 4, 8, 9, 10, 32, 37, 43, 49, 52, 62, 66, 70, 77, 79, 83, 88}
    # These cases were directly reviewed; all other samples stay coverage-only.
    validation_controls = {34, 67, 85}
    counts = collections.Counter()
    for kind, path in [('validation', validation), ('training', training)]:
        source_hash = file_digest(path)
        for line, row in enumerate(map(json.loads, path.open()), 1):
            if kind == 'training' and not row.get('repeat_reward_audit_path'):
                continue
            if not row.get('repeat_reward_audit_path'):
                raise ValueError('Missing native capture artifact')
            events, transcript, _, segments = replay_evidence(row, p)
            # Training additions cover EVERY primary positive quorum, including reviewer vetoes.
            with gzip.open(row['repeat_reward_audit_path'], 'rt') as stream:
                audit = json.load(stream)
            verdict = audit['verdict']
            if kind == 'training' and not verdict.get('evidence_review_called'):
                continue
            task = row['task_id']; part = partitions.setdefault(task, partition(task))
            cid = f'native:sl8nvath:{kind}:{line}'
            incomplete = (not row.get('evaluation_valid') or verdict.get('reason') == 'incomplete_public_context'
                          or bool(row.get('repeat_reward_actor_capture_failed')))
            expected = ('uncertain' if incomplete or kind == 'validation' and line in validation_controls
                        or kind == 'training' and line == 281
                        else 'repetitive' if kind == 'validation' and line in native_positives else None)
            reason = ('Incomplete/invalid original evaluation; a penalty is unsupported.' if incomplete else
                      'Audited bounded investigation, changing repair or source paging; protect against false penalties.' if expected == 'uncertain' else
                      'Directly audited sustained tool or actor-generation loop.' if expected == 'repetitive' else
                      'Coverage only; prior verdict or unsupported citations do not establish a whole-trajectory gold label.')
            pointer = {k: row[k] for k in ('repeat_reward_audit_path', 'repeat_reward_audit_sha256')}
            cases.append({'id': cid, 'task_key': task, 'partition': part, **pointer,
                          'raw': {'raw_score': row['raw_score']},
                          'parse_issues': ['incomplete_original_public_context'] if incomplete else [],
                          'source': {'run': 'sl8nvath', 'kind': kind, 'line': line, 'path': str(path.resolve()),
                                     'source_sha256': source_hash, 'row_sha256': digest(row)}})
            labels.append({'id': cid, 'expected': expected, 'reason': reason, 'scope': 'full_trajectory',
                           'annotation_status': 'unadjudicated' if expected is None else 'code_agent_reviewed'})
            counts[kind] += 1
            if kind == 'training' and line == 39:
                # The full trace could have another loop; isolate the false delimiter accusation.
                prefix = {'id': cid + ':boundary-control', 'task_key': task, 'partition': part,
                          'public_events': events[:4], 'actor_segments': [s for s in segments if s['after_event_id'] <= 4],
                          'public_context': '', 'raw': {'raw_score': 1},
                          'source': {'parent': cid, **pointer, 'view': 'audited_prefix'}}
                cases.append(prefix)
                labels.append({'id': prefix['id'], 'expected': 'uncertain', 'scope': 'full_trajectory',
                               'reason': 'Different actor investigation plans with intervening tools; recurring boundary tokens are not a loop.'})
    for name, events, segments, expected in contracts():
        cid = 'synthetic:trajectory-v13:' + name
        cases.append({'id': cid, 'task_key': cid, 'partition': partition(cid), 'public_events': events,
                      'actor_segments': segments, 'public_context': '', 'raw': {'raw_score': 1}})
        labels.append({'id': cid, 'expected': expected, 'scope': 'full_trajectory',
                       'annotation_status': 'unadjudicated' if expected is None else 'code_agent_reviewed',
                       'reason': 'Independent rule/chronology/quoted-data control: ' + name})
    if len({c['id'] for c in cases}) != len(cases) or {c['id'] for c in cases} != {x['id'] for x in labels}:
        raise ValueError('Fixture/label identity mismatch')
    groups = collections.defaultdict(set)
    for case in cases:
        groups[case['task_key']].add(case['partition'])
    if any(len(parts) != 1 for parts in groups.values()):
        raise ValueError('Task partition leakage')
    output.mkdir(parents=True)
    with (output / 'fixtures.jsonl.gz').open('wb') as binary:
        with gzip.GzipFile(fileobj=binary, mode='wb', mtime=0) as compressed:
            for case in cases:
                compressed.write((json.dumps(case, ensure_ascii=False, sort_keys=True) + '\n').encode())
    write_json(output / 'labels.json', {'policy_version': p.POLICY_VERSION, 'cases': labels,
               'reviewer': 'Code-agent evidence review; human adjudication pending. Protected uncertain is never automatically gold clean.'})
    result = {'policy_version': p.POLICY_VERSION, 'serializer_version': p.SERIALIZER_VERSION,
              'cases': len(cases), 'base_cases': manifest['cases'], 'native_additions': dict(counts),
              'label_counts': dict(collections.Counter(str(x['expected']) for x in labels)),
              'validation_sha256': file_digest(validation), 'training_sha256': file_digest(training),
              'base_manifest_sha256': file_digest(base / 'manifest.json'),
              'builder_sha256': file_digest(Path(__file__)), 'fixtures_sha256': file_digest(output / 'fixtures.jsonl.gz'),
              'labels_sha256': file_digest(output / 'labels.json')}
    write_json(output / 'manifest.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-suite', type=Path, default=Path('analysis/reward_model_trajectory_v12_20261015/frozen_task_disjoint'))
    parser.add_argument('--validation', type=Path, default=Path('validation_data/reward_model_train_v2/0.jsonl'))
    parser.add_argument('--training', type=Path, default=Path('rollout_data/reward_model_train_v2/1.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('analysis/reward_model_trajectory_v13_20261016/frozen_task_disjoint'))
    args = parser.parse_args()
    print(json.dumps(build(args.base_suite, args.validation, args.training, args.output), indent=2))


if __name__ == '__main__':
    main()
