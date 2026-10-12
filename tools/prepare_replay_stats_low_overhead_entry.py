"""Bind accepted helper math to reduced-overhead timers and post-timer audits.

Does not run CUDA. Keeps shared lightweight method/storage coverage, so this
is reduced instrumentation, not a claim of zero overhead. Held-out all28
fingerprints are captured in evaluation outside the quantization timer.
"""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def build_entry(raw,speed):
    anchor='\nimport hashlib\n'
    if raw.count(anchor)!=1:raise RuntimeError('raw accepted helper prefix changed')
    prefix=raw.split(anchor,1)[0]+'\n'
    if 'hessian_dead_subgroup_install' not in prefix or '_hinv_stable_expected=' not in prefix:
        raise RuntimeError('requires original stable/dead/workspace helper chain')
    start=raw.index('_replay_stats_installer_path=')
    end=raw.index('original_fwrd = impl.gptq_fwrd',start)
    stats=raw[start:end]
    if speed.count('original_fwrd = impl.gptq_fwrd')!=1:raise RuntimeError('timed forward boundary changed')
    speed=speed.replace('original_fwrd = impl.gptq_fwrd',stats+'\noriginal_fwrd = impl.gptq_fwrd')
    start=raw.index('_layer_trace_original_eval = eval_utils.kl_ppl_eval')
    end=raw.index('sys.argv[0] = "ptq.py"',start)
    fingerprints=raw[start:end]
    # Preserve accepted fresh Stage0 and forbid stage2 recomputation. Layer
    # fingerprints execute only during evaluation after the main speed timer.
    guard='''def _forbid_precompute(*args, **kwargs):
    raise RuntimeError('timing stage2 must consume its verified fresh cache')
if '--exit_after_precompute' not in sys.argv:
    impl.collect_static_end_to_end_saliency_and_fisher=_forbid_precompute
'''
    if speed.count('sys.argv[0] = "ptq.py"')!=1:raise RuntimeError('speed runpy boundary changed')
    speed=speed.replace('sys.argv[0] = "ptq.py"',fingerprints+'\n'+guard+'\nsys.argv[0] = "ptq.py"')
    speed=speed.replace('"layer_output_capture": False','"layer_output_capture_during_quantization": False,\n              "heldout_layer_output_capture_after_timer": True,\n              "shared_cpu_method_storage_observers_retained": True')
    result=prefix+speed
    compile(result,'replay_stats_low_overhead_entry','exec')
    return result


def prepare_entry():
    return build_entry((ROOT/'outputs/replay-stats-full-followup-20261012-v2/entry.py').read_text(),
                       (ROOT/'tools/realq_low_overhead_speed_entry.py').read_text())
