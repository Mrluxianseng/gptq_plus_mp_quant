"""Serial follow-up, only after complete independent longer-input admission."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_qa_memory_rows_completed import sha
from tools.storage_method_log_stream import LiveFileRelay


def main():
    os.chdir(ROOT)
    root = ROOT / 'outputs/qa-lmhead-night-followup-20261014-v1'
    root.mkdir(exist_ok=False)
    source = [Path(__file__), ROOT/'tools/run_qa_lm_head_component.py',
        ROOT/'tools/qa_lm_head_storage_prototype.py', ROOT/'tools/run_qa_layerwise_component_v2.py',
        ROOT/'tools/run_qa_layerwise_component.py', ROOT/'tools/qa_layerwise_weight_prototype.py',
        ROOT/'tools/qa_scoring_cache_offload_prototype.py', ROOT/'tools/qa_score_lifetime_prototype.py']
    bindings = {str(p): sha(p) for p in source}
    (root/'source_bindings.json').write_text(json.dumps(bindings, indent=2))
    admission = ROOT/'outputs/qa-layerwise-local-qwen17-ctx256-20261014-v2/independent_complete_kv_audit.json'
    print('CPU waiting for complete independent ctx256 admission: '+str(admission), flush=True)
    deadline = time.monotonic() + 3600
    while not admission.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError('Admission not completed within one hour; no GPU launched')
        time.sleep(10)
    if json.loads(admission.read_text()).get('valid') is not True:
        raise RuntimeError('Independent admission rejected; no GPU launched')
    completed = []
    for model in ('Qwen3-0.6B', 'Qwen3-1.7B'):
        if any(sha(p) != digest for p, digest in bindings.items()):
            raise RuntimeError('Frozen follow-up source changed')
        tag = 'qwen06' if model == 'Qwen3-0.6B' else 'qwen17'
        output = Path('outputs')/f'qa-lmhead-local-{tag}-ctx128-20261014-v1'
        command = [sys.executable, '-u', 'tools/run_qa_lm_head_component.py',
            '--model', '/mnt/d/llamaModels/'+model, '--output', str(output), '--length', '128']
        logfile = root/(tag+'.log')
        with logfile.open('x') as stream:
            proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            relay = LiveFileRelay(logfile)
            while proc.poll() is None:
                relay.forward()
                time.sleep(1)
            relay.forward()
        independent = output/'independent_completed_audit.json'
        audit_command = [sys.executable, 'tools/run_qa_lm_head_component.py',
            '--audit-root', str(output), '--audit-output', str(independent)]
        with (root/(tag+'_audit.log')).open('x') as stream:
            audited = subprocess.run(audit_command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                stdout=stream, stderr=subprocess.STDOUT)
        record = dict(model=model, output=str(output), returncode=proc.returncode,
            independent_returncode=audited.returncode,
            independent_valid=independent.exists() and json.loads(independent.read_text()).get('valid') is True)
        completed.append(record)
        if not record['independent_valid'] or record['returncode'] != 0 or record['independent_returncode'] != 0:
            with (root/'status.json').open('x') as stream:
                json.dump(dict(valid=False, completed=completed, admission_sha256=sha(admission)), stream, indent=2)
            raise SystemExit('Head method or infrastructure rejected; all original artifacts retained')
    with (root/'status.json').open('x') as stream:
        json.dump(dict(valid=True, completed=completed, admission_sha256=sha(admission)), stream, indent=2)


if __name__ == '__main__':
    main()
