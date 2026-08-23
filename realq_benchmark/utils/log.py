"""Logging setup for RealQ. Thin wrapper over utils.log_utils."""
import os

from utils import log_utils


def init(output_dir: str, exp: str) -> str:
    """Initialise process-wide logging. Returns the log directory."""
    log_dir = os.path.join(output_dir, exp)
    log_utils.init_logging(log_dir)
    return log_dir
