"""Forward growing log files without pipes, GPU work, or changing raw evidence."""
from __future__ import annotations

from pathlib import Path
import sys


class LiveFileRelay:
    def __init__(self, path, destination=None, *, progress_lines=True):
        self.path = Path(path)
        self.destination = destination if destination is not None else sys.stdout.buffer
        self.progress_lines = progress_lines
        self.offset = 0

    def forward(self):
        try:
            stream = self.path.open('rb')
        except FileNotFoundError:
            return 0
        forwarded = 0
        with stream:
            stream.seek(0, 2)
            if stream.tell() < self.offset:
                raise RuntimeError('Log was truncated while following: ' + str(self.path))
            stream.seek(self.offset)
            while True:
                data = stream.read(65536)
                if not data:
                    break
                self.offset += len(data)
                forwarded += len(data)
                if self.progress_lines:
                    data = data.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
                self.destination.write(data)
                self.destination.flush()
        return forwarded
