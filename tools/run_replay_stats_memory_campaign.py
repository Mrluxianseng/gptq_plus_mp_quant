"""Campaign adapter registering only the audited storage-lifetime switch.

Original comparator and workers remain unchanged. All algorithm/data settings
are still compared, alongside state, output, cache and resource checks.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import compare_full_pipeline_profiles as audit

ENGINEERING_FLAG = "--research_replay_early_stats"


def normalized_command(command):
    original = audit.SYSTEM_COMMAND_FLAGS
    try:
        audit.SYSTEM_COMMAND_FLAGS = original | {ENGINEERING_FLAG}
        return audit.algorithm_command(command)
    finally:
        audit.SYSTEM_COMMAND_FLAGS = original


def main():
    audit.SYSTEM_COMMAND_FLAGS.add(ENGINEERING_FLAG)
    from tools.run_full_pipeline_memory_campaign import main as campaign_main
    campaign_main()


if __name__ == "__main__":
    main()
