"""Bounded read-only telemetry; begins mid-baseline, not complete coverage."""
import json
from pathlib import Path
import subprocess
import time

root = Path('/mnt/d/gptq_plus_realq/outputs/resume_20260927_1216')
end = time.monotonic() + 3600
with (root / 'gpu_telemetry.jsonl').open('x') as out:
    while time.monotonic() < end:
        active = [p.name for p in root.iterdir() if p.is_dir()
                  and (p/'manifest.json').exists() and not (p/'status.json').exists()]
        result = subprocess.run([
            '/usr/lib/wsl/lib/nvidia-smi',
            '--query-gpu=memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.draw',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
        out.write(json.dumps({'time':time.strftime('%F %T'), 'active_arms':active,
                              'returncode':result.returncode,
                              'memory_MiB_util_pct_temp_C_clock_MHz_power_W':result.stdout.strip(),
                              'stderr':result.stderr.strip()})+'\n')
        out.flush()
        if (root/'cross-corrected/status.json').exists():
            break
        time.sleep(10)
