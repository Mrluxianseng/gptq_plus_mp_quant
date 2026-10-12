"""One append-only progress view over serial research controllers.

Reads existing authoritative worker/campaign logs; launches no GPU jobs and
never controls their lifetime. Includes source paths and completed status.
"""
import argparse
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queues", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--append", action="store_true", help="Resume an ended progress view without replacing its history")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    positions = {}
    seen_active = {}
    seen_status = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a" if args.append else "x", encoding="utf-8") as destination:
        def write(text):
            destination.write(text)
            destination.flush()
            print(text, end="", flush=True)
        write(f"\n[progress {time.strftime('%F %T')}] {'Resuming' if args.append else 'Starting'} serial research view.\n")
        while True:
            all_finished = True
            for queue in args.queues:
                active = queue / "active.json"
                if active.exists():
                    try:
                        current = json.loads(active.read_text())
                    except json.JSONDecodeError:
                        current = None  # controller may be rewriting this file
                    if current is not None:
                        label = current["label"]
                        if seen_active.get(queue) != label:
                            write(f"\n[progress {time.strftime('%F %T')}] queue={queue.name} active={label}\n")
                            seen_active[queue] = label
                        command = current["command"]
                        sources = [queue / f"{label}.log"]
                        if "--tag" in command:
                            tag = command[command.index("--tag") + 1]
                            # Child stdout already mirrors combined.log; use
                            # only one so each line is forwarded once.
                            combined = root / "outputs" / f"memory_campaign_{tag}" / "combined.log"
                            if combined.exists():
                                sources = [combined]
                            elif any(value.endswith("run_full_pipeline_memory_profile.py") for value in command):
                                # Direct capacity profiles do not have a paired
                                # campaign combined.log; their driver only logs
                                # summaries, so forward each raw stage log too.
                                run = root / "outputs" / f"full_pipeline_profile_{tag}"
                                sources += [run / "stage0.log", run / "stage2.log"]
                        for source in sources:
                            if source.exists():
                                with source.open(encoding="utf-8", errors="replace") as stream:
                                    stream.seek(positions.get(source, 0))
                                    text = stream.read()
                                    positions[source] = stream.tell()
                                if text:
                                    write(f"\n[log-source {source}]\n{text}")
                status = queue / "status.json"
                if not status.exists():
                    all_finished = False
                else:
                    text = status.read_text()
                    if seen_status.get(queue) != text:
                        write(f"\n[controller-status {queue.name}]\n{text}\n")
                        seen_status[queue] = text
            if all_finished:
                write(f"\n[progress] All listed controllers completed at {time.strftime('%F %T')}; inspect individual validity flags.\n")
                break
            time.sleep(1)


if __name__ == "__main__":
    main()
