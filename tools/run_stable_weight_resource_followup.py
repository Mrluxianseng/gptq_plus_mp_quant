"""Continue a numerically correct resource failure with a new solver schedule.

Wait until the preceding GPU queue exits; run a fresh two-model component
preflight before any actual trajectory. Keep strict gates and all failures.
"""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_full_pipeline_memory_campaign import stop_process_tree


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary-queue',type=Path,required=True)
    p.add_argument('--output-directory',type=Path,required=True)
    p.add_argument('--fallback-directory',type=Path,required=True)
    p.add_argument('--preflight-directory',type=Path,required=True)
    p.add_argument('--tag-suffix',required=True)
    args=p.parse_args()
    out=args.output_directory
    out.mkdir(exist_ok=False)
    child=reader=None
    report=dict(valid=False,next_method_started=False,method='distinct two-buffer solver out= schedule plus stable weight; no parameter changes')

    def interrupt(sig,frame):
        raise KeyboardInterrupt(f'signal {sig}')

    signal.signal(signal.SIGTERM,interrupt)

    def execute(label,command,mirror=None):
        nonlocal child
        (out/'active.json').write_text(json.dumps(dict(label=label,command=command)))
        print('[workspace-continuation] starting '+label,flush=True)
        with (out/(label+'.log')).open('x') as stream:
            child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            while child.poll() is None:
                if mirror and (mirror/'active.json').exists():
                    try:
                        text=(mirror/'active.json').read_text()
                        json.loads(text)
                        (out/'active.json').write_text(text)
                    except (OSError,json.JSONDecodeError):
                        pass
                time.sleep(2)
        return child.returncode

    try:
        print('[workspace-continuation] waiting for preceding queue; no GPU process',flush=True)
        while not (args.primary_queue/'status.json').exists():
            time.sleep(5)
        parent=read(args.primary_queue/'status.json')
        report['primary_status']=parent
        if parent.get('valid'):
            report.update(valid=True,note='Primary requested gates completed; no redundant alternative launched. Independent audit remains separate.')
            return
        failures=[]
        for tag in parent.get('campaigns',[]):
            for file in (ROOT/'outputs'/('memory_campaign_'+tag)).glob('pair*_audit/comparison.json'):
                j=read(file)
                if not j.get('strict_measured_memory_gate_passed'):
                    failures.append(dict(path=str(file),exact=j.get('exactness_passed'),
                        phase_regressions=j.get('phase_regressions'),boundary_regressions=j.get('boundary_regressions')))
        if not failures or not all(f['exact'] for f in failures):
            report['note']='No complete exact resource failure; inspect numerical/infrastructure bug rather than mask it by a blind method switch'
            return
        report.update(trigger_audits=failures,next_method_started=True,
            reflection='Stable final weight destination did not satisfy every measured memory interval. Next isolate solver allocation schedule: same three solver APIs, two reused buffers, preserving factor strides; no private pool.')
        (out/'decision.json').write_text(json.dumps(report,indent=2))
        with (ROOT/'docs/memory_speed_optimization_journal.md').open('a',encoding='utf-8') as f:
            f.write('\n\n### AS条件续接：既有权重存储方案资源失败后继续工作区验证\n\n'
                    '**理论与参考。** OLLA的最后消费者/内存位置思想（https://arxiv.org/html/2210.12924v2）；'
                    'PyTorch2.11 cholesky与cholesky_inverse的out=接口。\n\n'
                    '**触发证据及具体方法。** 完成的数值正确资源失败见'+str(out/'decision.json')+'。'
                    '保留原三步求逆和dtype/stride：H_work最后被Cholesky读取后用作逆矩阵目标；'
                    'chol最后被inverse读取后用作最终upper因子目标，每步输入与输出存储不同。'
                    '随后保留消费H组复用和既有权重写回，不启用私有池。原inplace damping分支回到原helper，实际目标轨迹不启用该分支。\n\n'
                    '**评价及反思。** 这是新分配调度的小实验，尚无正向结果。先实际CUDA两模型原值/回退/布局核对；'
                    '精确失败保留并禁止上真实轨迹；成功后才fresh layer0–2，再过资源门槛才28层三对。'
                    '不剔除失败，不改校准/Adam参数，诊断计时不算加速。\n')
        # Preceding reader sees its terminal status before this writer starts.
        time.sleep(5)
        reader=subprocess.Popen([sys.executable,'-u',str(ROOT/'tools/stream_memory_research_progress.py'),
            '--queues',str(out),'--output',str(ROOT/'logs/realq-system-restart-20261010-1408.log'),'--append'],
            cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        command=[sys.executable,'-u',str(ROOT/'tools/check_hessian_solver_workspace_cuda.py'),
                 '--output-directory',str(args.preflight_directory),'--model-paths',
                 '/mnt/d/llamaModels/Qwen3-0.6B','/mnt/d/llamaModels/Qwen3-1.7B']
        code=execute('solver-preflight',command)
        report['preflight_returncode']=code
        status=args.preflight_directory/'status.json'
        if code or not status.exists() or not read(status).get('valid'):
            report['note']='New solver placement failed CUDA preflight; no actual quantization launched. Inspect recorded error before further implementation.'
            return
        report['preflight_status']=read(status)
        command=[sys.executable,'-u',str(ROOT/'tools/run_hessian_solver_workspace_followup.py'),
                 '--preflight',str(status),'--output-directory',str(args.fallback_directory),
                 '--tag-suffix',args.tag_suffix,'--models','qwen17b','qwen06b',
                 '--prefix-stop-layer','2','--full-pairs','3']
        report['returncode']=execute('solver-controller',command,mirror=args.fallback_directory)
        if (args.fallback_directory/'status.json').exists():
            report['fallback_status']=read(args.fallback_directory/'status.json')
            report['valid']=bool(report['fallback_status'].get('valid'))
        if report['valid']:
            command=[sys.executable,'-u',str(ROOT/'tools/audit_hessian_solver_workspace_followup.py'),
                     '--queue',str(args.fallback_directory),'--output-directory',str(out/'independent-audit'),
                     '--tag-suffix',args.tag_suffix,'--variant','workspace-stable','--models','qwen17b','qwen06b']
            code=execute('independent-audit',command)
            report['independent_audit']=read(out/'independent-audit/status.json')
            report['valid']=not code and bool(report['independent_audit'].get('valid'))
    except BaseException as exc:
        if child is not None and child.poll() is None:
            stop_process_tree(child.pid)
            child.wait()
        report['error']=repr(exc)
        raise
    finally:
        report['finished']=time.strftime('%F %T')
        (out/'status.json').write_text(json.dumps(report,indent=2))
        if reader is not None:
            reader.wait(timeout=15)
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
