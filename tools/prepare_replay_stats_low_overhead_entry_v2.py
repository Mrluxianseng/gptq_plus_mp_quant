"""Repair only the post-timer fingerprint destination binding.

The original frozen generator and failed GPU evidence stay unchanged.
Quantization, timers, helper math and calibration are byte-for-byte preserved.
"""
from tools.prepare_replay_stats_low_overhead_entry import ROOT,build_entry as original_build_entry


def build_entry(raw,speed):
    source=original_build_entry(raw,speed)
    old='(root / "layer_output_fingerprints.json")'
    if source.count(old)!=1:raise RuntimeError('heldout fingerprint output boundary changed')
    source=source.replace(old,'(destination / "layer_output_fingerprints.json")')
    compile(source,'bound_post_timer_fingerprint_destination','exec')
    return source


def prepare_entry():
    return build_entry((ROOT/'outputs/replay-stats-full-followup-20261012-v2/entry.py').read_text(),
                       (ROOT/'tools/realq_low_overhead_speed_entry.py').read_text())
