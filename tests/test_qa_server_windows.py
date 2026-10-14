import json
import os
from pathlib import Path
import subprocess
import sys
import time


def test_host_window_graceful_stop_preserves_real_samples_and_summary(tmp_path):
    if sys.platform!='linux':return
    output=tmp_path/'host.jsonl';stop=tmp_path/'stop.json'
    root=Path(__file__).resolve().parents[1]
    child=subprocess.Popen([sys.executable,str(root/'tools/run_qa_host_window.py'),
        '--pid',str(os.getpid()),'--output',str(output),'--stop',str(stop)],cwd=root)
    try:
        deadline=time.monotonic()+12
        while time.monotonic()<deadline:
            if output.exists() and len(output.read_text().splitlines())>=2:break
            time.sleep(0.1)
        assert output.exists() and output.stat().st_size>0
        stop.write_text('{}')
        assert child.wait(timeout=8)==0
        row=json.loads(output.with_suffix('.summary.json').read_text())
        assert row['completed'] and row['complete_samples']>=1 and row['sources_unchanged']
        assert row['termination']=='explicit_QA_window_closed'
    finally:
        if child.poll() is None:child.kill();child.wait()


def test_full_formal_entry_requires_direct_identity_and_dedicated_telemetry():
    from tools.run_storage_formal_qa_mapped_v4 import source
    from tools.run_qa_server_mapped_component_v3 import source as component
    from tools.audit_qa_server_mapped_v2 import phases
    text=source();compile(text,'actual_server_formal_v4_test','exec')
    assert 'audit_qa_server_mapped_v3' in text and 'qa_server_mapped_telemetry' in text
    assert 'readmit(root)' in text
    worker=component();compile(worker,'actual_server_component_v3_test','exec')
    assert worker.index('dist_utils.distribute_model(model)')<worker.index('before={n:id(p)')
    assert len(phases(4,['synthetic_interface']))==593
    assert len(phases(4,list(range(10))))==611
