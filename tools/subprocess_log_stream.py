"""Forward subprocess output promptly, including carriage-return progress bars."""

from __future__ import annotations

import codecs
from collections.abc import Iterator
from typing import TextIO


def iter_output_chunks(stream: TextIO, chunk_size: int = 8192) -> Iterator[str]:
    """Yield decoded child output as it arrives, normalizing progress `\r` to lines.

    ``TextIOWrapper`` iteration waits for newline characters. Many progress bars
    update with bare carriage returns, so they otherwise remain invisible in a
    redirected/nohup log until a later newline or process exit.
    """
    binary = stream.buffer
    decoder = codecs.getincrementaldecoder(stream.encoding or "utf-8")(errors="replace")
    pending_cr = False
    while True:
        raw = binary.read1(chunk_size)
        if not raw:
            break
        decoded = decoder.decode(raw)
        if pending_cr:
            if decoded.startswith("\n"):
                decoded = decoded[1:]
            yield "\n"
            pending_cr = False
        decoded = decoded.replace("\r\n", "\n")
        if decoded.endswith("\r"):
            decoded = decoded[:-1]
            pending_cr = True
        decoded = decoded.replace("\r", "\n")
        if decoded:
            yield decoded
    tail = decoder.decode(b"", final=True)
    if pending_cr:
        yield "\n"
    if tail:
        yield tail.replace("\r\n", "\n").replace("\r", "\n")
