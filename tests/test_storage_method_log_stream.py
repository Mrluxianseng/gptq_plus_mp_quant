"""Verify live output, tail draining, and following a running four-arm plan."""
import io
import json
import os
from pathlib import Path
import sys

import psutil
import pytest

from tools.storage_method_log_stream import LiveFileRelay
from tools import run_storage_method_pipeline as pipeline
from tools import stream_storage_method_formal as observer


def test_relay_missing_growing_utf8_progress_and_no_duplicates(tmp_path):
    path, output = tmp_path/'stage.log', io.BytesIO()
    relay = LiveFileRelay(path, output)
    assert relay.forward() == 0
    text = 'Quantizing Layers: 27/28\r最后一层完成\n'.encode()
    path.write_bytes(text[:-2])
    relay.forward()
    with path.open('ab') as file:
        file.write(text[-2:]+b'last error\n')
    relay.forward()
    assert relay.forward() == 0
    assert output.getvalue() == text.replace(b'\r', b'\n')+b'last error\n'
    assert path.read_bytes() == text+b'last error\n'
    path.write_bytes(b'x')
    with pytest.raises(RuntimeError, match='truncated'):
        relay.forward()


@pytest.mark.parametrize('exit_code', [0, 7])
def test_real_cpu_worker_stage_stdout_stderr_and_final_tail(tmp_path, capsys, exit_code):
    # Actual subprocess/file sampling/relay path; no GPU or torch execution.
    command = [sys.executable, '-u', '-c',
               "import sys,time; print('stage worker starting',flush=True); "
               "time.sleep(.35); sys.stdout.write('27/28\\r28/28\\n'); "
               "sys.stderr.write('worker final stderr\\n'); "
               "sys.stdout.flush(); sys.stderr.flush(); sys.exit("+str(exit_code)+")"]
    class RssWriter:
        def __init__(self): self._file = io.StringIO()
        def writerow(self, row): self._file.write(json.dumps(row)+'\n')
    path = tmp_path/'stage.log'
    result = pipeline.run_stage('stage2', command, tmp_path, dict(os.environ, PYTHONUNBUFFERED='1'), path, 30, psutil, RssWriter())
    output = capsys.readouterr().out
    assert result['returncode'] == exit_code
    assert 'stage worker starting' in output and 'worker final stderr' in output
    assert '27/28\n28/28\n' in output
    assert output.count('stage worker starting') == 1
    assert b'27/28\r28/28\n' in path.read_bytes()


def test_legacy_four_arm_observer_follows_new_stages_without_restarting(tmp_path, monkeypatch):
    jobs = []
    for tag in ('q06-control', 'q06-candidate', 'q17-control', 'q17-candidate'):
        result = tmp_path/tag
        result.mkdir()
        jobs.append(dict(tag=tag, result=str(result)))
    (tmp_path/'plan.json').write_text(json.dumps({'jobs': jobs}))
    controller_log = tmp_path/'nohup.log'
    controller_log.write_text('Plan and controller starting\n')
    (tmp_path/'q06-control.log').write_text('pipeline alive\n')
    (Path(jobs[0]['result'])/'stage0.log').write_text('Fisher 10/256\rFisher 256/256\n')
    output = io.BytesIO()
    steps = [0]
    def progress(_):
        steps[0] += 1
        assert steps[0] <= 2
        for job in jobs:
            result = Path(job['result'])
            for stage in ('stage0', 'stage2'):
                if steps[0] == 1 and job == jobs[0] and stage == 'stage0':
                    continue
                with (result/(stage+'.log')).open('a') as file:
                    file.write(f"{job['tag']} {stage} wave{steps[0]}\n")
        if steps[0] == 2:
            (tmp_path/'status.json').write_text(json.dumps({'valid': False, 'error': 'fixture failure retained'}))
    monkeypatch.setattr(observer.time, 'sleep', progress)
    result = observer.run(tmp_path, destination=output, controller_log=controller_log)
    text = output.getvalue().decode()
    assert result['valid'] is False
    assert text.count('Fisher 10/256\nFisher 256/256') == 1
    assert text.count('pipeline alive') == 1
    assert text.count('Plan and controller starting') == 1
    for job in jobs:
        for stage in ('stage0', 'stage2'):
            assert f"{job['tag']} {stage} wave2" in text
    assert 'fixture failure retained' in text


def test_observer_does_not_duplicate_new_aggregate_plan(tmp_path):
    (tmp_path/'plan.json').write_text(json.dumps({'aggregate_worker_stdout': True}))
    with pytest.raises(ValueError, match='already forwards'):
        observer.run(tmp_path, follow_once=True)
