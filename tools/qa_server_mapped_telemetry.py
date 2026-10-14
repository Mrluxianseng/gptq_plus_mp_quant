"""Separate all-device QA NVML sampling and read-only process-tree telemetry."""
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from tools import qa_server_mapped_validation as ancestor


class TelemetryServerBridge(ancestor.ServerBridge):
    def observed_eval(self,model,tokenizer,*args,**kwargs):
        # Native dispatch precedes this boundary and may reconstruct Parameters.
        # Require unchanged object identities throughout QA, in both arms.
        before={name:id(p) for name,p in model.named_parameters(remove_duplicate=False)}
        completed=False
        try:
            result=super().observed_eval(model,tokenizer,*args,**kwargs)
            completed=True
            return result
        finally:
            after={name:id(p) for name,p in model.named_parameters(remove_duplicate=False)}
            groups={}
            for name,value in before.items():groups.setdefault(value,[]).append(name)
            row=dict(completed=completed and before==after,parameter_identity_during_QA=before==after,
                parameter_names=list(before),shared_parameter_groups=sorted(sorted(names) for names in groups.values()),
                scope='Identities from after original dispatch through QA; pre-dispatch control replacement is separate')
            with (self.root/'qa_parameter_identity_rank0.json').open('x') as f:json.dump(row,f,indent=2)
            if completed and not row['completed']:raise RuntimeError('Parameter identities changed during QA')

    def qa_eval(self,model,tokenizer,*args,**kwargs):
        root=self.root;root.mkdir(parents=True,exist_ok=True)
        started=time.time();monitors=[]
        nvml_path=root/'qa_all_devices_nvml.csv'
        manifest=root/'qa_telemetry_window.json'
        if nvml_path.exists() or manifest.exists():raise FileExistsError('Preserve old QA telemetry')
        completed=False
        with nvml_path.open('x') as nvml,(root/'qa_host_monitor.log').open('x') as log:
            try:
                monitors.append(subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,index,uuid',
                    '--format=csv,noheader,nounits','--loop-ms=100'],stdout=nvml,stderr=log))
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
                monitors.append(subprocess.Popen([sys.executable,'-u','tools/run_qa_host_window.py',
                    '--pid',str(os.getpid()),'--output',str(root/'qa_host_tree.jsonl'),'--stop',str(root/'qa_host_stop.json')],
                    stdout=log,stderr=subprocess.STDOUT,env=env))
                result=super().qa_eval(model,tokenizer,*args,**kwargs)
                completed=True
                return result
            finally:
                with (root/'qa_host_stop.json').open('x') as f:json.dump(dict(end_unix=time.time()),f)
                if monitors and monitors[0].poll() is None:monitors[0].terminate()
                for child in monitors:
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                with manifest.open('x') as f:json.dump(dict(completed=completed,pid=os.getpid(),
                    start_unix=started,end_unix=time.time(),monitor_exit_codes=[p.returncode for p in monitors],
                    nvml_is_sampled=True,host_is_sampled_non_atomic=True,
                    scope='Dedicated QA sampling window including storage cleanup; no retroactive peak attribution'),f,indent=2)


text=inspect.getsource(ancestor.install)
anchor='return ServerBridge('
if text.count(anchor)!=1:raise RuntimeError('Frozen bridge construction changed')
scope=dict(ancestor.__dict__,ServerBridge=TelemetryServerBridge)
exec(compile(text,'qa_server_telemetry_install','exec'),scope)
install=scope['install']
