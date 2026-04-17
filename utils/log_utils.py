import logging
import sys
import os
import datetime
from contextlib import contextmanager


class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: '\033[94m',       # Blue
        logging.INFO: '\033[92m',        # Green
        logging.WARNING: '\033[93m',     # Yellow
        logging.ERROR: '\033[91m',       # Red
        logging.CRITICAL: '\033[1;91m',  # Bold Red
    }
    RESET = '\033[0m'

    def format(self, record):
        # Get the color for the specific log level
        color = self.COLORS.get(record.levelno, self.RESET)
        
        # Save the original format string
        format_orig = self._style._fmt
        
        # Inject the color code at the start, and the RESET code right before the message
        self._style._fmt = f"{color}[%(asctime)s | %(levelname)s]{self.RESET} %(message)s"
        
        # Format the record
        result = super().format(record)
        
        # Restore the original format string
        self._style._fmt = format_orig
        
        return result


def init_logging(log_dir):
    """
    Initializes logging to output colored prefixes to the console
    and plain text to a timestamped file.

    DP: when launched via torchrun, only LOCAL_RANK==0 gets the full logger.
    Other ranks are silenced to CRITICAL level so we don't see every message
    twice (and don't fight each other for the same log file). If a rank>0
    actually raises an error it will still surface via stderr and
    torchrun's aggregation.
    """
    # Detect torchrun rank. This env var is only set under distributed launch.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    is_main = (rank == 0)

    # 1. Create log directory (only rank 0 writes to it).
    if is_main:
        os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    # 2. Define formats
    log_format = '[%(asctime)s | %(levelname)s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    if not is_main:
        # Non-main ranks: nuke existing handlers, suppress everything below CRITICAL.
        # Keep CRITICAL so unrecoverable errors still reach stderr.
        logging.basicConfig(
            level=logging.CRITICAL,
            handlers=[logging.NullHandler()],
            force=True,
        )
        logging.disable(logging.ERROR)
        return

    # 3. Configure Console Handler (WITH color prefix)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter(log_format, datefmt=date_format))

    # 4. Configure File Handler (WITHOUT color, plain text)
    log_file_path = os.path.join(log_dir, f"log_{timestamp}.txt")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    # 5. Apply handlers
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler],
        force=True  # Overwrites any existing logging configuration
    )


def set_logging_enabled(enabled: bool, level=logging.INFO):
    """
    Enables or disables logging globally.
    """
    if enabled:
        logging.disable(logging.NOTSET)
        logging.getLogger().setLevel(level)
    else:
        logging.disable(logging.CRITICAL)


@contextmanager
def disable_logging_context(highest_level=logging.CRITICAL):
    """
    A context manager to temporarily disable logging.
    
    Args:
        highest_level: All logs at or below this level will be suppressed.
                       Defaults to logging.CRITICAL (suppresses everything).
    """
    # Save the current disable level so we can restore it later
    previous_level = logging.root.manager.disable
    
    # Disable logging
    logging.disable(highest_level)
    
    try:
        yield
    finally:
        # Guarantee that logging is restored even if an exception occurs
        logging.disable(previous_level)


# --- Usage Example ---
if __name__ == "__main__":
    init_logging("logs")
    
    logging.info("This message is white, but the prefix is green!")
    logging.warning("This message is white, but the prefix is yellow!")
    logging.error("This message is white, but the prefix is red!")
