import hashlib
import os
from pathlib import Path
from tools.qa_host_memory_observer_v2 import bounded_sha,HostTreeObserver,collect
from test_qa_host_memory_observer import process


def test_bounded_hash_and_complete_actual_cli_accounting_without_reading_full_log(tmp_path,monkeypatch):
    path=tmp_path/'large.bin'
    chunk=bytes(range(256))*4096
    with path.open('wb') as stream:
        for _ in range(8):stream.write(chunk)
    expected=hashlib.sha256(chunk*8).hexdigest()
    original=Path.read_bytes
    def no_full_log_read(self):
        if self.name in ('large.bin','record.jsonl'):raise AssertionError('Do not read full telemetry files into memory')
        return original(self)
    monkeypatch.setattr(Path,'read_bytes',no_full_log_read)
    assert bounded_sha(path)==expected
    result=collect(os.getpid(),tmp_path/'record.jsonl',interval=0.1,seconds=0.2)
    assert result['complete_samples']>0 and result['source_unchanged'] and result['ancestor_source_unchanged']
    assert result['runtime_identity']['boot_id'] and result['jsonl_sha256']==bounded_sha(tmp_path/'record.jsonl')
    assert result['formal_enabled'] is False


def test_known_boot_change_refused_and_unknown_not_fabricated(tmp_path):
    process(tmp_path,11,1,30,50)
    observer=HostTreeObserver(11,tmp_path,tmp_path/'unused')
    assert observer.boot_id is None and observer.runtime['boot_id'] is None
    boot=tmp_path/'sys/kernel/random/boot_id';boot.parent.mkdir(parents=True);boot.write_text('boot-one')
    observer=HostTreeObserver(11,tmp_path,tmp_path/'unused')
    boot.write_text('boot-two')
    row=observer.sample()
    assert not row['complete'] and row['termination']=='boot_identity_changed'
