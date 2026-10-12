"""Fresh full comparison removing only a rejected input-cache allocation change."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_full_pipeline_memory_campaign import stop_process_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(exist_ok=False)
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    old = ROOT / "outputs/full-memory-restart-20261010-1408/active.json"
    command = json.loads(old.read_text())["command"]
    command[0] = sys.executable
    command[command.index("--tag") + 1] = args.tag
    folder = out / "manifests"
    folder.mkdir()
    for side in ("control", "candidate"):
        for stage in ("stage0", "stage2"):
            flag = f"--{side}-{stage}-manifest"
            source = Path(command[command.index(flag) + 1])
            data = json.loads(source.read_text())
            if side == "candidate" and stage == "stage2":
                if data["command"].count("--stage2_input_cache_cpu_init") != 1:
                    raise RuntimeError("expected exactly one rejected input allocation flag")
                data["command"].remove("--stage2_input_cache_cpu_init")
            data["scope_note"] = "Fresh all28 quantized; shared original GPU input-cache init/offload_inps and microbatch2. Candidate only whole Stage0 checkpoint + CE256."
            destination = folder / f"{side}-{stage}.json"
            destination.write_text(json.dumps(data, indent=2))
            command[command.index(flag) + 1] = str(destination)
    (out / "active.json").write_text(json.dumps(dict(label=args.tag, command=command), indent=2))
    proc = None
    result = dict(started=time.strftime("%F %T"), removed_only="--stage2_input_cache_cpu_init",
                  fresh_statistics=True, reuse_incomplete_cache=False)
    try:
        print(f"[checkpoint-ce] starting {args.tag}", flush=True)
        with (out / f"{args.tag}.log").open("x") as log:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            proc.wait()
        result.update(returncode=proc.returncode, valid=proc.returncode == 0)
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            stop_process_tree(proc.pid)
            proc.wait()
        result.update(valid=False, error=repr(exc))
        raise
    finally:
        result["finished"] = time.strftime("%F %T")
        (out / "status.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
