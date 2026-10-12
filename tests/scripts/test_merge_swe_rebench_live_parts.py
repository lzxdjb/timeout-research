"""A running filter's closed parts can form a checked, independent snapshot."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_swe_rebench_reference_filter import live_fixture, new_live, wait_until

from scripts import merge_swe_rebench_live_parts as merge


@pytest.fixture
def published(tmp_path):
    source, audit_dir, cases, states, signature = live_fixture(tmp_path)
    writer = new_live(source, audit_dir, cases, {}, signature, part_rows=1, flush_seconds=.02)
    for case in reversed(cases):
        key = case['inventory']['id']
        writer.submit(key, states[key]['qualified_sha256'])
    def ready():
        path = audit_dir / 'live_summary.json'
        return path.exists() and json.loads(path.read_text()).get('parquet_rows') == 2
    wait_until(ready, writer, timeout=15)
    try:
        yield source, audit_dir, writer
    finally:
        writer.close()


def test_merge_while_writer_is_alive_preserves_rows_schema_and_proofs(tmp_path, published):
    source, audit_dir, writer = published
    output = tmp_path / 'snapshot.parquet'
    receipt = merge.merge_snapshot(audit_dir, output)
    assert writer.thread.is_alive()
    assert receipt['rows'] == receipt['parts'] == 2
    assert receipt['provisional'] and not receipt['final_filter_checks_complete']
    merged = pq.read_table(output)
    summary = json.loads((audit_dir / 'live_summary.json').read_text())
    expected = pa.concat_tables([pq.read_table(p['path']) for p in summary['parts']])
    assert merged.equals(expected, check_metadata=True)
    assert merged.schema.equals(pq.read_schema(source), check_metadata=True)
    with pytest.raises(ValueError, match='exists'):
        merge.merge_snapshot(audit_dir, output)
    # An unfinished/orphan part without a receipt is not in the frozen summary.
    (audit_dir / 'live_parquet_parts/part-999999.parquet').write_bytes(b'incomplete')
    assert merge.merge_snapshot(audit_dir, tmp_path / 'later.parquet')['rows'] == 2


@pytest.mark.parametrize('fault', ['part', 'proof', 'duplicate', 'schema', 'rebuilding'])
def test_merge_rejects_corrupt_or_unqualified_snapshot(tmp_path, published, fault):
    _, audit_dir, writer = published
    writer.close()
    summary_path = audit_dir / 'live_summary.json'
    summary = json.loads(summary_path.read_text())
    part = Path(summary['parts'][0]['path'])
    if fault == 'part':
        part.write_bytes(b'corrupt')
    elif fault == 'proof':
        receipt = json.loads(part.with_suffix('.parquet.audit.json').read_text())
        proof = audit_dir / 'cases' / receipt['checkpoints'][0]['id'] / 'qualified_record.json'
        proof.write_text('{}')
    elif fault == 'duplicate':
        summary['parts'].append(summary['parts'][0])
    elif fault == 'schema':
        pq.write_table(pa.table({'bad': [1]}), part)
        sha = merge.audit.file_sha256(part)
        summary['parts'][0]['output_sha256'] = sha
        receipt_path = part.with_suffix('.parquet.audit.json')
        receipt = json.loads(receipt_path.read_text())
        receipt['output_sha256'] = sha
        merge.audit.audit_write_json(receipt_path, receipt)
    else:
        summary['rebuilding'] = True
    merge.audit.audit_write_json(summary_path, summary)
    with pytest.raises(ValueError):
        merge.merge_snapshot(audit_dir, tmp_path / 'bad.parquet')
    assert not (tmp_path / 'bad.parquet').exists()
