"""Independent exact native/private address and output-pointer accounting."""
import hashlib
import json
from pathlib import Path


def check_row(row, recorded):
    if any(row.get(key) != value for key, value in recorded.items()):
        return False
    identity = tuple(row['pool_id'])
    private = row['private_segments']
    device = int(row['output_device'].split(':')[1])
    native = [s for s in row['native_segments'] if s['device'] == device]
    if not private or private != [s for s in native if tuple(s['segment_pool_id']) == identity]:
        return False
    for segment in native:
        address = segment['address']
        for block in segment['blocks']:
            if block['address'] != address:
                return False
            address += block['size']
        if address != segment['address'] + segment['total_size']:
            return False
    allocated = sum(b['size'] for s in native for b in s['blocks'] if b['state'] == 'active_allocated')
    reserved = sum(s['total_size'] for s in native)
    private_reserved = sum(s['total_size'] for s in private)
    active = sum(s['active_size'] for s in private)
    pointer = row['output_data_ptr']
    payload = row['output_payload_bytes']
    covered = any(b['state'] == 'active_allocated' and b['address'] <= pointer
                  and pointer + payload <= b['address'] + b['size']
                  for s in private for b in s['blocks'])
    return bool(covered and row['accounting_complete'] and row['callback_error'] == 0
        and row['private_segments_in_native']
        and row['current_device'] == device
        and allocated == row['native_allocated'] == row['snapshot_native_allocated']
        and reserved == row['native_reserved'] == row['snapshot_native_reserved']
        and private_reserved == row['private_reserved'] == row['mapped_host_live']
        and active == row['private_active'] and active >= payload
        and row['callback_allocations'] - row['callback_releases'] == len(private))


def audit(root):
    root = Path(root)
    journal = root/'qa_head_accounting_journal.jsonl'
    record = json.loads((root/'qa_mapped_head_rank0.json').read_text())
    head = record.get('head', {})
    originals = head.get('records', [])
    digest = hashlib.sha256()
    count = 0
    failures = []
    with journal.open('rb') as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            if count >= len(originals) or row['call'] != count+1 or not check_row(row, originals[count]):
                failures.append(count+1)
            count += 1
    valid = bool(record['completed'] and count == len(originals) == head['calls'] and count > 0
        and not failures and len(head['batch_releases']) == count
        and head['final_host_live'] == 0 and head['allocations'] == head['releases'])
    return dict(valid=valid, records=count, failed_calls=failures, journal_sha256=digest.hexdigest(),
        scope='Independent exact raw segment/block/payload pointers and original head records; no QA equivalence substitute')
