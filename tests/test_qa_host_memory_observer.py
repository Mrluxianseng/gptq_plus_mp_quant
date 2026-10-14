import mmap
import os
from pathlib import Path
import pytest
from tools.qa_host_memory_observer import kb_fields,identity,HostTreeObserver,collect,FIELDS


def process(proc,pid,parent,ticks,pss):
    folder=proc/str(pid);folder.mkdir()
    # stat comm may contain spaces and right parentheses.
    tail=['S',str(parent)]+['0']*17+[str(ticks)]+['0']*3
    (folder/'stat').write_text(str(pid)+' (name ) with spaces) '+' '.join(tail))
    (folder/'smaps_rollup').write_text('\n'.join(f'{key}: {pss if key=="Pss" else 100} kB' for key in FIELDS))
    (folder/'status').write_text('VmRSS: 100 kB\nVmLck: 0 kB\n')
    (folder/'cgroup').write_text('0::/job\n')


def test_parse_units_names_and_complete_tree_pss_without_rss_substitution(tmp_path):
    proc=tmp_path/'proc';proc.mkdir();cg=tmp_path/'cgroup';(cg/'job').mkdir(parents=True)
    (cg/'job/memory.max').write_text('85899345920')
    (cg/'job/memory.current').write_text('2000')
    (proc/'meminfo').write_text('MemTotal: 1000 kB\nMemAvailable: 900 kB\nSwapTotal: 200 kB\nSwapFree: 100 kB\n')
    process(proc,11,1,30,50);process(proc,12,11,40,50);process(proc,20,1,50,99)
    observer=HostTreeObserver(11,proc,cg);row=observer.sample()
    assert row['complete'] and len(row['processes'])==2 and not row['atomic_snapshot']
    assert row['totals_bytes']['Rss']==200*1024 and row['totals_bytes']['Pss']==100*1024
    assert row['processes'][0]['status_bytes']['VmPin'] is None
    assert row['capacity']['cgroup_v2_ancestors'][0]['values']['memory.max']==85899345920
    assert row['capacity']['cgroup_v2_ancestors'][1]['unavailable']['memory.max']=='FileNotFoundError'
    assert identity(proc,11)['start_ticks']==30
    with pytest.raises(ValueError):kb_fields('Pss: 1 kB\nPss: 2 kB')
    with pytest.raises(ValueError):kb_fields('Rss: 1 kB',('Pss',))


def test_missing_smaps_and_pid_reuse_remain_incomplete(tmp_path):
    process(tmp_path,11,1,30,50);process(tmp_path,12,11,40,50)
    observer=HostTreeObserver(11,tmp_path,tmp_path/'unused')
    (tmp_path/'12/smaps_rollup').unlink()
    row=observer.sample()
    assert not row['complete'] and row['errors'] and len(row['processes'])==1
    assert row['totals_bytes']['Pss']==50*1024 # Explicit partial aggregate, never filled with RSS.
    p=tmp_path/'11/stat';p.write_text(p.read_text().replace(' 30 ',' 31 '))
    assert observer.sample()['termination']=='root_exited_or_identity_changed'


def test_actual_linux_shared_mapping_reduces_pss_double_count_and_fresh_output(tmp_path):
    size=32*1024*1024
    mapping=mmap.mmap(-1,size,flags=mmap.MAP_SHARED)
    for offset in range(0,size,4096):mapping[offset]=1
    ready_r,ready_w=os.pipe();stop_r,stop_w=os.pipe()
    child=os.fork()
    if child==0:
        os.close(ready_r);os.close(stop_w)
        for offset in range(0,size,4096):assert mapping[offset]==1
        os.write(ready_w,b'1');os.read(stop_r,1);os._exit(0)
    try:
        os.close(ready_w);os.close(stop_r)
        assert os.read(ready_r,1)==b'1'
        row=HostTreeObserver(os.getpid()).sample()
        assert row['complete'] and {p['identity']['pid'] for p in row['processes']}=={os.getpid(),child}
        assert row['totals_bytes']['Rss']-row['totals_bytes']['Pss']>=size
        summary=collect(os.getpid(),tmp_path/'real.jsonl',interval=0.1,seconds=0.2)
        assert summary['complete_samples']>0 and summary['source_unchanged']
        with pytest.raises(FileExistsError):collect(os.getpid(),tmp_path/'real.jsonl',seconds=0.2)
    finally:
        os.write(stop_w,b'1');os.close(stop_w);os.close(ready_r);os.waitpid(child,0);mapping.close()
