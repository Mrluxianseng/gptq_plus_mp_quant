"""Prepared exact QA allocator intervals, with explicit device phase identity.

Caller must ensure exclusive ownership of peak-counter resets during QA. Idle
devices are queried without synchronize/reset/context creation. NVML sampling
and whole-process integration remain separate obligations.
"""
import json
import os
from pathlib import Path
import time


class VisibleDevicePhases:
    def __init__(self, cuda, destination, *, exclusive_peak_owner, rank=0,
                 prior_process_peak_getter=None):
        if exclusive_peak_owner is not True:
            raise RuntimeError('Exact profiling requires exclusive QA peak-counter ownership')
        self.cuda = cuda
        self.count = cuda.device_count()
        if self.count <= 0:
            raise RuntimeError('Requires at least one visible CUDA device')
        self.root = Path(destination)
        self.root.mkdir(parents=True, exist_ok=False)
        self.rank = int(rank)
        self.files = []
        self.paths = []
        self.used = set()
        self.closed = False
        self.phase = 'qa_setup'
        self.started = time.perf_counter()
        self.start = self.snapshot()
        self.peaks = [[row['peak_allocated'], row['peak_reserved']] for row in self.start]
        if prior_process_peak_getter is not None:
            for index, row in enumerate(self.peaks):
                prior = prior_process_peak_getter(f'cuda:{index}')
                row[:] = [max(row[0],int(prior[0])),max(row[1],int(prior[1]))]
        for index in range(self.count):
            folder = self.root/f'device{index}'
            folder.mkdir()
            path = folder/f'cuda_memory_evaluation_rank{self.rank}_pid{os.getpid()}.jsonl'
            self.paths.append(str(path))
            self.files.append(path.open('x',encoding='utf-8'))
        self.reset_used()

    def snapshot(self):
        rows = []
        for index in range(self.count):
            def get():
                return dict(allocated=int(self.cuda.memory_allocated(index)),
                    reserved=int(self.cuda.memory_reserved(index)),
                    peak_allocated=int(self.cuda.max_memory_allocated(index)),
                    peak_reserved=int(self.cuda.max_memory_reserved(index)))
            row = get()
            if any(row.values()):
                self.used.add(index)
                self.cuda.synchronize(index)
                row = get()
            rows.append(row)
        return rows

    def reset_used(self):
        for index in sorted(self.used):
            self.cuda.reset_peak_memory_stats(index)

    def mark(self, next_phase):
        if self.closed:
            raise RuntimeError('Profiler already closed')
        end = self.snapshot()
        for index, (before, after, stream) in enumerate(zip(self.start,end,self.files)):
            peak_a = max(before['allocated'],after['allocated'],after['peak_allocated'])
            peak_r = max(before['reserved'],after['reserved'],after['peak_reserved'])
            self.peaks[index] = [max(self.peaks[index][0],peak_a),max(self.peaks[index][1],peak_r)]
            row = dict(phase=f'qa_device{index}__{self.phase}', logical_phase=self.phase,
                next_phase=f'qa_device{index}__{next_phase}', device_index=index,
                rank=self.rank, stage='stage2', logkind='evaluation',
                peak_measurement_mode='exact', start_allocated_bytes=before['allocated'],
                start_reserved_bytes=before['reserved'],end_allocated_bytes=after['allocated'],
                end_reserved_bytes=after['reserved'],sampled_peak_allocated_bytes=peak_a,
                sampled_peak_reserved_bytes=peak_r,
                cumulative_peak_allocated_bytes=self.peaks[index][0],
                cumulative_peak_reserved_bytes=self.peaks[index][1],
                elapsed_seconds=time.perf_counter()-self.started)
            stream.write(json.dumps(row)+'\n');stream.flush()
        self.phase = str(next_phase)
        self.start = end
        self.started = time.perf_counter()
        self.reset_used()

    def finish(self, status):
        if status not in ('completed','failed'):
            raise RuntimeError('Explicit completed/failed final status required')
        if self.closed:
            raise RuntimeError('Profiler already closed')
        self.mark('qa_complete' if status=='completed' else 'qa_failed')
        for stream in self.files: stream.close()
        self.closed = True
        selectors = os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
        devices = [dict(device_index=index,
            physical_selector=selectors[index] if index<len(selectors) and selectors[index] else str(index),
            peak_allocated_bytes=row[0],peak_reserved_bytes=row[1]) for index,row in enumerate(self.peaks)]
        report = dict(status=status,device_peak_schema='all-visible-device-maxima-v1',
            visible_device_count=self.count,devices=devices,
            peak_allocated_bytes=max(row[0] for row in self.peaks),
            peak_reserved_bytes=max(row[1] for row in self.peaks),memory_profile_files=self.paths,
            scope='QA exact allocator intervals plus explicitly supplied prior process maxima; not NVML')
        with (self.root/'qa_device_phase_summary.json').open('x') as stream:
            json.dump(report,stream,indent=2)
        return report
