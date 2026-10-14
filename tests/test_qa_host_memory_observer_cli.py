import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def test_actual_standalone_entry_from_another_working_directory(tmp_path):
    output=tmp_path/'observed.jsonl'
    subprocess.run([sys.executable,str(ROOT/'tools/run_qa_host_memory_observer.py'),
        '--pid','self','--output',str(output),'--seconds','0.2','--interval','0.1'],cwd=tmp_path,
        check=True,capture_output=True,text=True)
    launch=json.loads(output.with_suffix('.launch.json').read_text())
    assert launch['completed'] and launch['sources_unchanged'] and not launch['formal_enabled']
    assert launch['summary']['complete_samples']>0 and launch['summary']['runtime_identity']['boot_id']
    assert all('Pss' in row['totals_bytes'] for row in map(json.loads,output.read_text().splitlines()))


def test_actual_missing_pid_preserves_false_without_fabricated_samples(tmp_path):
    output=tmp_path/'missing.jsonl'
    result=subprocess.run([sys.executable,str(ROOT/'tools/run_qa_host_memory_observer.py'),
        '--pid','2147483647','--output',str(output),'--seconds','0.2'],cwd=tmp_path,
        capture_output=True,text=True)
    assert result.returncode!=0
    launch=json.loads(output.with_suffix('.launch.json').read_text())
    assert not launch['completed'] and launch['error_type']=='FileNotFoundError'
    assert not output.exists() and not output.with_suffix('.summary.json').exists()
