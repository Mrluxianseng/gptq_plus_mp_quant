import json
import os
from pathlib import Path
import select
import subprocess
import sys
from tools.qa_host_memory_observer_v3 import HostTreeObserver
from test_qa_host_memory_observer import process

ROOT=Path(__file__).resolve().parents[1]


def test_reparented_root_stays_observed_but_pid_reuse_refused(tmp_path):
    process(tmp_path,11,1,30,50);process(tmp_path,12,11,40,50)
    observer=HostTreeObserver(11,tmp_path,tmp_path/'unused')
    path=tmp_path/'11/stat';path.write_text(path.read_text().replace('S 1 ','S 99 '))
    row=observer.sample()
    assert row['complete'] and len(row['processes'])==2
    assert row['processes'][0]['identity']['ppid']==99 and observer.root_identity['ppid']==1
    path.write_text(path.read_text().replace(' 30 ',' 31 '))
    assert observer.sample()['termination']=='root_exited_or_identity_changed'


def test_real_linux_parent_exit_does_not_stop_monitor():
    ready_r,ready_w=os.pipe();exit_r,exit_w=os.pipe();stop_r,stop_w=os.pipe()
    launcher=os.fork()
    if launcher==0:
        os.close(ready_r);os.close(exit_w);os.close(stop_w)
        worker=os.fork()
        if worker==0:
            os.close(ready_w);os.close(exit_r)
            os.read(stop_r,1);os._exit(0)
        os.write(ready_w,str(worker).encode());os.read(exit_r,1);os._exit(0)
    worker=None
    try:
        os.close(ready_w);os.close(exit_r);os.close(stop_r)
        assert select.select([ready_r],[],[],5)[0]
        worker=int(os.read(ready_r,64))
        observer=HostTreeObserver(worker)
        assert observer.root_identity['ppid']==launcher
        os.write(exit_w,b'1');os.waitpid(launcher,0);launcher=None
        row=observer.sample()
        assert row['complete'] and row['processes'][0]['identity']['ppid']!=observer.root_identity['ppid']
        assert row['processes'][0]['identity']['start_ticks']==observer.root_identity['start_ticks']
    finally:
        os.write(stop_w,b'1');os.close(stop_w);os.close(exit_w);os.close(ready_r)
        if launcher is not None:os.waitpid(launcher,0)


def test_actual_cli_from_external_directory_and_all_source_provenance(tmp_path):
    output=tmp_path/'stable.jsonl'
    subprocess.run([sys.executable,str(ROOT/'tools/run_qa_host_memory_observer_v2.py'),
        '--pid','self','--output',str(output),'--seconds','0.2','--interval','0.1'],
        cwd=tmp_path,check=True,capture_output=True,text=True)
    launch=json.loads(output.with_suffix('.launch.json').read_text())
    assert launch['completed'] and launch['sources_unchanged'] and len(launch['source_sha256'])==5
    assert launch['summary']['observer_v2_source_unchanged'] and launch['summary']['ancestor_source_unchanged']
    assert launch['summary']['complete_samples']>0 and not launch['formal_enabled']
