"""Bounded head-only allocator diagnostic, never formal QA acceptance.

Replays recorded shapes, not tokens or quantized model values. The frozen
accounting guard still executes. Its raw failed record is flushed before
exception cleanup, with both current and output-device counters.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def shapes(path):
    rows = []
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('kind') == 'model_input':
                if row['forward'] != len(rows) + 1 or len(row['shape']) != 2:
                    raise RuntimeError('Recorded input sequence incomplete')
                b, length = row['shape']
                if not 1 <= b <= 16 or length <= 0:
                    raise RuntimeError('Invalid recorded natural batch geometry')
                rows.append(row)
    if not rows:
        raise RuntimeError('No real input geometry recorded')
    return rows


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--max-calls', type=int, default=257)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = shapes(args.events)[:args.max_calls]
    from tools.qa_server_mapped_validation import verify_runtime
    from tools.qa_mapped_head_storage_prototype import prepare_runtime
    from tools.qa_mapped_head_natural_batch import NaturalBatchMappedHeadStorage
    from tools.qa_server_runtime_compat import SnapshotCUDA
    import torch
    binary, bindings = verify_runtime(args.runtime_root)
    library, capabilities = prepare_runtime(binary)
    cuda = SnapshotCUDA(torch.cuda)
    allocator = torch.cuda.memory.CUDAPluggableAllocator(
        str(binary), 'qa_mapped_alloc', 'qa_mapped_free')
    torch.cuda.set_device(args.device)
    head = torch.nn.Linear(1024, 151936, bias=False, dtype=torch.bfloat16,
                           device=f'cuda:{args.device}').eval()
    source_paths = [Path(__file__), Path('tools/qa_mapped_head_storage_prototype.py'),
                    Path('tools/qa_mapped_head_batch_lifetime.py'),
                    Path('tools/qa_mapped_head_natural_batch.py'),
                    Path('tools/qa_server_runtime_compat.py')]
    report = dict(scope='Head-only recorded-shape allocator diagnostic; synthetic values, no QA acceptance',
                  pid=os.getpid(), input_count=len(inputs), events_sha256=digest(args.events),
                  bindings=bindings, source_sha256={str(p): digest(p) for p in source_paths},
                  capabilities=capabilities, completed=False, guard_passed=False)
    with (args.output/'manifest.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    journal = (args.output/'head_records.jsonl').open('x', buffering=1)

    class CapturedHead(NaturalBatchMappedHeadStorage):
        def post(self, module, values, output):
            before = len(self.records)
            try:
                return super().post(module, values, output)
            finally:
                if len(self.records) > before:
                    row = dict(self.records[-1])
                    row['current_device'] = torch.cuda.current_device()
                    row['output_device'] = None if output is None else str(output.device)
                    row['last_error'] = int(library.qa_last_error())
                    row['devices'] = [dict(index=i,
                        allocated=torch.cuda.memory_allocated(i),
                        reserved=torch.cuda.memory_reserved(i)) for i in range(torch.cuda.device_count())]
                    row['callback_allocations'] = int(library.qa_allocation_count())
                    row['callback_releases'] = int(library.qa_release_count())
                    row['private_snapshot'] = self.pool.snapshot(include_traces=False)
                    journal.write(json.dumps(row) + '\n')
                    journal.flush()

    ticket = CapturedHead(head, enabled=True, expected_shape=(151936, 1024),
                         library=library, cuda=cuda,
                         pool_factory=lambda: cuda.MemPool(allocator.allocator()))
    started = time.monotonic()
    # Catch outside the call stack so exception frames cannot retain the mapped
    # output during pool teardown. Do not call release_batch on a failed call.
    def run():
        with torch.inference_mode():
            for row in inputs:
                value = torch.zeros((*row['shape'], 1024), device=f'cuda:{args.device}',
                                    dtype=torch.bfloat16)
                output = head(value)
                torch.cuda.synchronize(args.device)
                del output, value
                ticket.release_batch()
    try:
        run()
        report['guard_passed'] = True
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
    gc.collect()
    torch.cuda.synchronize(args.device)
    if report['guard_passed']:
        report['head'] = ticket.close()
        report['completed'] = True
    else:
        for handle in ticket.handles:
            handle.remove()
        ticket.handles = []
        if ticket.context is not None:
            ticket.context.__exit__(None, None, None)
            ticket.context = None
        if ticket.pool is not None:
            snap = ticket.pool.snapshot(include_traces=False)
            active = sum(segment['active_size'] for segment in snap)
            report['failed_pool_active_after_unwind'] = active
            if active == 0:
                ticket.pool = None
                gc.collect()
                torch.cuda.synchronize(args.device)
        report['unaccepted_head_records'] = ticket.records
    report.update(wall_seconds=time.monotonic()-started,
                  callback_live=int(library.qa_live_bytes()),
                  callback_peak=int(library.qa_peak_bytes()),
                  callback_allocations=int(library.qa_allocation_count()),
                  callback_releases=int(library.qa_release_count()),
                  journal_sha256=digest(args.output/'head_records.jsonl'))
    journal.close()
    with (args.output/'status.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k: report.get(k) for k in ('completed','guard_passed','error',
        'wall_seconds','callback_live','failed_pool_active_after_unwind')}), flush=True)
    # Never unload the callback library while a live allocator segment survives.
    if report['callback_live']:
        os._exit(2)
    if not report['guard_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
