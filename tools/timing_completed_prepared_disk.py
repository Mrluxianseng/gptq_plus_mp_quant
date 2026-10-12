"""Lossless disk links only after a complete independently audited timing arm.

All original paths and byte contents remain; fresh Fisher/statistics are
never linked. This is disk housekeeping outside all GPU timing intervals.
"""
import json
import os
from pathlib import Path
import stat
import time
import uuid

from tools.deduplicate_completed_prepared_checkpoint import digest, filesystem_path

ROOT = Path(__file__).resolve().parents[1]


def link_completed(destination, reference, *, workspace=None):
    workspace = filesystem_path(workspace or ROOT/'outputs').resolve(strict=True)
    runs = [filesystem_path(p).resolve(strict=True) for p in (reference, destination)]
    if any(p.is_symlink() or not p.is_relative_to(workspace) for p in runs):
        raise ValueError('completed disk link must stay inside workspace outputs')
    ref, target = runs
    state = json.loads((ref/'status.json').read_text())
    timed = json.loads((target/'status.json').read_text())
    extra = json.loads((target/'independent_complete_method_and_output_audit.json').read_text())
    if (not state.get('valid') or state.get('scope') != 'full_pipeline'
            or state['stage0']['returncode'] != 0 or state['stage2']['returncode'] != 0
            or not timed.get('numerical_reference_validated')
            or not timed.get('fresh_static_cache_exact_to_reference')
            or not all(extra.get(k) is True for k in ('valid', 'all28_outputs_exact',
                'method_records_exact', 'post_quantization_cache_exact', 'outside_speed_timers'))):
        raise ValueError('complete reference and timing numerical/method/cache audit required')
    metrics, quality = timed['metrics_by_rank'], timed['quality_by_rank']
    if (len(metrics) != 1 or metrics[0]['state_tensors'] != 507
            or metrics[0]['state_sha256'] != state['metrics']['state_sha256']
            or quality != [[dict(kl=state['kl'], ppl=state['ppl'])]]):
        raise ValueError('timed complete state/quality differs from reference')
    files = []
    for run in runs:
        matches = list((run/'static_cache/_prepared_checkpoints').glob('*/model.safetensors'))
        if len(matches) != 1 or matches[0].is_symlink() or not matches[0].resolve().is_relative_to(workspace):
            raise ValueError('exactly one contained prepared model required')
        files.append(matches[0])
    anchor, destination_file = files
    if anchor.parent.name != destination_file.parent.name:
        raise ValueError('prepared checkpoint identities differ')
    initial = [(p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    hashes = [digest(p) for p in files]
    if (initial[0][0] != initial[1][0] or hashes[0] != hashes[1]
            or initial != [(p.stat().st_size, p.stat().st_mtime_ns) for p in files]):
        raise ValueError('prepared bytes differ or changed while hashing')
    already = os.path.samefile(anchor, destination_file)
    if not already:
        temporary = destination_file.with_name(destination_file.name+'.link-'+uuid.uuid4().hex)
        try:
            os.link(anchor, temporary)
            if not os.path.samefile(anchor, temporary):
                raise RuntimeError('link does not refer to verified anchor')
            try:
                os.replace(temporary, destination_file)
            except PermissionError:
                if os.name != 'nt' or destination_file.stat().st_mode & stat.S_IWRITE:
                    raise
                old_mode = destination_file.stat().st_mode
                os.chmod(destination_file, stat.S_IREAD | stat.S_IWRITE)
                try:
                    os.replace(temporary, destination_file)
                except BaseException:
                    os.chmod(destination_file, old_mode)
                    raise
        finally:
            if temporary.exists():
                if os.name == 'nt':
                    os.chmod(temporary, stat.S_IREAD | stat.S_IWRITE)
                temporary.unlink()
            os.chmod(anchor, stat.S_IREAD)
    if not os.path.samefile(anchor, destination_file) or digest(destination_file) != hashes[0]:
        raise RuntimeError('installed disk link verification failed')
    return dict(valid=True, finished=time.strftime('%F %T'), reference=str(anchor),
                destination=str(destination_file), sha256=hashes[0], bytes=initial[0][0],
                already_linked=already, duplicate_bytes_released=0 if already else initial[0][0],
                all_paths_and_bytes_preserved=True, fresh_statistics_linked=False,
                scope='Completed prepared model disk storage only, outside GPU timers; no GPU memory contribution')
