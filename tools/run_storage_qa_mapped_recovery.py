"""Recover only the failed 0.6B candidate against its completed control.

All old artifacts/sources remain immutable. Original commands/settings/cache
are reused. Only the reviewed snapshot reader, durable diagnostic journal,
and an explicit verified quantized recovery checkpoint are added.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_storage_method_formal as formal
from tools.run_storage_formal_qa_mapped_v8 import entry as prior_entry
from tools.qa_server_mapped_validation import verify_runtime, sha


def entry(original):
    text = prior_entry(original)
    old = 'from tools.qa_server_mapped_telemetry import install as _qa_server_install'
    if text.count(old) != 1:
        raise RuntimeError('Original telemetry installation changed')
    text = text.replace(old, 'from tools.qa_server_mapped_telemetry_v2 import install as _qa_server_install')
    anchor = "runpy.run_path('ptq.py', run_name='__main__')"
    checkpoint = """if _qa_early_selected and _hinv_release:
    from tools.qa_quantized_recovery_checkpoint import install as _recovery_install
    _recovery_install(_qa_server_eval, Path(os.environ['QA_RECOVERY_QMODEL_PATH']))
"""
    if text.count(anchor) != 1:
        raise RuntimeError('Actual ptq entry changed')
    text = text.replace(anchor, checkpoint + '\n' + anchor)
    compile(text, 'safe_snapshot_formal_recovery', 'exec')
    return text


def verify_diagnostic(root):
    root = Path(root)
    row = formal.read(root/'status.json')
    if not row['completed'] or not row['guard_passed'] or row['input_count'] != 514:
        raise RuntimeError('Safe snapshot repeated-head diagnostic missing')
    if row['callback_live'] != 0 or row['callback_allocations'] != row['callback_releases']:
        raise RuntimeError('Diagnostic pool release incomplete')
    if not all(r['exact'] for r in row['numerical_probes']) or len(row['numerical_probes']) != 7:
        raise RuntimeError('Diagnostic complete GEMM byte checks missing')
    for path, expected in {**row['bindings'], **row['source_sha256']}.items():
        if sha(path) != expected:
            raise RuntimeError('Frozen diagnostic/runtime source changed: ' + path)
    journal = root/'head_records.jsonl'
    if sha(journal) != row['journal_sha256']:
        raise RuntimeError('Original diagnostic journal changed')
    rows = [json.loads(line) for line in journal.open()]
    if len(rows) != 514 or any(r['call'] != i+1 or not r['accounting_complete'] for i, r in enumerate(rows)):
        raise RuntimeError('Every diagnostic head accounting record required')
    return row


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--prior-root', type=Path, required=True)
    p.add_argument('--runtime-root', type=Path, required=True)
    p.add_argument('--diagnostic-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    prior = args.prior_root.resolve()
    plan = formal.read(prior/'plan.json')
    control = next(j for j in plan['jobs'] if j['model'] == 'qwen06b' and j['side'] == 'control')
    failed = next(j for j in plan['jobs'] if j['model'] == 'qwen06b' and j['side'] == 'candidate')
    control_root = Path(control['result'])
    if not formal.read(control_root/'status.json')['valid'] or formal.read(Path(failed['result'])/'status.json')['valid']:
        raise RuntimeError('Requires the original completed control and preserved failed candidate')
    bindings = formal.read(prior/'source_bindings.json')
    formal.check_sources(bindings)
    diagnostic = verify_diagnostic(args.diagnostic_root)
    binary, runtime_bindings = verify_runtime(args.runtime_root)
    # Preserve every older executed source/library/cache binding. New sources
    # join a new receipt rather than rewriting the v4 admission manifest.
    old_entry = Path(failed['launcher'][failed['launcher'].index('--stage2-manifest')+1])
    descriptor = formal.read(old_entry)
    old_command = descriptor['command']
    old_worker = next(Path(x) for x in old_command if str(x).endswith('/entry.py'))
    new_worker = args.output/'entry.py'
    new_worker.write_text(entry(formal.build_entry()), encoding='utf-8')
    inherited = formal.read(args.runtime_root/(prior.name+'_admitted_sources.json'))
    for path, expected in inherited.items():
        if sha(path) != expected:
            raise RuntimeError('Inherited formal QA source changed: ' + path)
    extra = [Path(__file__), ROOT/'tools/qa_server_runtime_compat_v2.py',
             ROOT/'tools/qa_server_mapped_telemetry_v2.py', ROOT/'tools/qa_quantized_recovery_checkpoint.py',
             ROOT/'tools/run_qa_mapped_accounting_diagnostic.py',
             ROOT/'tools/run_qa_mapped_accounting_diagnostic_v2.py',
             ROOT/'tools/run_qa_mapped_accounting_diagnostic_v3.py']
    inherited.update({str(path.resolve()): sha(path) for path in extra})
    qa_bindings_path = args.output/'qa_admitted_sources.json'
    formal.save(qa_bindings_path, inherited)
    bindings.update({str(path.relative_to(ROOT)): sha(path) for path in extra})
    formal.save(args.output/'source_bindings.json', bindings)
    tag = args.output.name + '-qwen06b-candidate'
    new_manifests = []
    for option in ('--stage0-manifest', '--stage2-manifest'):
        source = Path(failed['launcher'][failed['launcher'].index(option)+1])
        data = formal.read(source)
        data['command'] = [str(new_worker.resolve()) if value == str(old_worker) else value for value in data['command']]
        destination = args.output/(option[2:]+'.json')
        formal.save(destination, data)
        new_manifests.append(destination)
    launcher = list(failed['launcher'])
    for option, destination in zip(('--stage0-manifest', '--stage2-manifest'), new_manifests):
        launcher[launcher.index(option)+1] = str(destination.resolve())
    launcher[launcher.index('--tag')+1] = tag
    cache = formal.read(prior/'cache_ready_qwen06b.json')['directory']
    launcher += ['--reuse-cache', cache]
    result_root = ROOT/'outputs'/('full_pipeline_profile_'+tag)
    checkpoint = args.output/'qwen06b_quantized_recovery.pt'
    env = dict(os.environ, **plan['environment'], REALQ_QA_STORAGE_VALIDATION='1',
        QA_MAPPED_HEAD_LIBRARY=str(binary.resolve()), QA_MAPPED_HEAD_LIBRARY_SHA256=sha(binary),
        QA_SERVER_RUNTIME_ROOT=str(args.runtime_root.resolve()), QA_SERVER_BINDINGS=str(qa_bindings_path.resolve()),
        QA_RECOVERY_QMODEL_PATH=str(checkpoint.resolve()), HF_DATASETS_CACHE=plan['qa_datasets_cache'],
        HF_DATASETS_OFFLINE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONPATH=str(ROOT))
    receipt = dict(control=str(control_root), failed_candidate=failed['result'], candidate=str(result_root),
        launcher=launcher, original_stage2_command=old_command,
        recovery_stage2_command=formal.read(new_manifests[1])['command'],
        source_worker_sha256=sha(old_worker), recovery_worker_sha256=sha(new_worker),
        checkpoint=str(checkpoint), diagnostic_status_sha256=sha(args.diagnostic_root/'status.json'),
        controller_pid=os.getpid(), executed=False,
        scope='Incremental storage ablation recovery, NOT unoptimized baseline or stable speed benchmark')
    formal.save(args.output/'plan.json', receipt)
    if not args.run:
        print(json.dumps(receipt, indent=2)); return
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU compute process exists; refuse concurrent launch')
    status = dict(valid=False, completed=False, executed=True, candidate=str(result_root), control=str(control_root))
    formal.save(args.output/'launch.json', dict(pid=os.getpid(), time_unix=time.time(), launcher=launcher))
    try:
        print('Starting repaired candidate only; existing control preserved: '+str(control_root), flush=True)
        subprocess.run(launcher, cwd=ROOT, env=env, check=True)
        formal.check_sources(bindings)
        if sha(new_worker) != receipt['recovery_worker_sha256']:
            raise RuntimeError('Recovery worker changed')
        from tools.audit_storage_formal_global_capacity import audit_pair
        from tools.audit_qa_server_mapped_v3 import audit
        from utils.eval_utils import PAPER_QA_TASKS
        pair = audit_pair(control_root, result_root, 4, bindings)
        qa = audit(control_root, result_root, PAPER_QA_TASKS, inherited, expected_devices=4)
        pair['qa_whole_mapped_storage'] = qa
        pair['checks']['complete_exact_server_qa'] = qa['valid']
        pair['checks']['verified_recovery_checkpoint'] = formal.read(checkpoint.with_suffix('.pt.receipt.json'))['completed']
        pair['valid'] = pair['capacity_valid'] = all(pair['checks'].values())
        pair['strict_interval_point_valid'] &= qa['strict_local_nonincrease']
        formal.save(args.output/'independent_complete_pair_audit.json', pair)
        status.update(valid=pair['valid'], completed=True)
        if not pair['valid']:
            raise RuntimeError('Formal recovery pair rejected; preserved complete audit')
    except BaseException as exc:
        status.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        status['finished_unix'] = time.time()
        formal.save(args.output/'status.json', status)


if __name__ == '__main__':
    main()
