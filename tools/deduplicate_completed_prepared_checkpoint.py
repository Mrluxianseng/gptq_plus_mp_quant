"""Lossless disk housekeeping for two completed immutable model preparations.

This does not reduce GPU memory and does not share fresh Fisher statistics.
Validates workspace containment, completed status, streamed byte hashes and
stable file metadata before atomically installing a same-volume hard link.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import time
import uuid


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            value.update(chunk)
    return value.hexdigest()


def filesystem_path(path):
    """Permit verified Windows experiment paths and temporary link names >260 chars."""
    absolute = os.path.abspath(path)
    if os.name == "nt" and not absolute.startswith("\\\\?\\"):
        absolute = ("\\\\?\\UNC\\" + absolute[2:]) if absolute.startswith("\\\\") else "\\\\?\\" + absolute
    return Path(absolute)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control-run", type=Path, required=True)
    p.add_argument("--candidate-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    workspace = filesystem_path(Path(__file__).resolve().parents[1] / "outputs")
    files = []
    for directory in (args.control_run, args.candidate_run):
        directory = filesystem_path(directory).resolve(strict=True)
        if not directory.is_relative_to(workspace.resolve()):
            raise ValueError("run is outside workspace outputs")
        status = json.loads((directory / "status.json").read_text())
        if not status.get("valid") or status["stage0"]["returncode"] != 0:
            raise ValueError("run is not completed and numerically valid")
        paths = list((directory / "static_cache/_prepared_checkpoints").glob("*/model.safetensors"))
        if len(paths) != 1:
            raise ValueError("requires exactly one prepared model shard")
        path = paths[0]
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(workspace.resolve()):
            raise ValueError("prepared model path leaves workspace")
        files.append(path)
    control, candidate = files
    if control.parent.name != candidate.parent.name:
        raise ValueError("prepared model identities differ")
    initial = [(path.stat().st_size, path.stat().st_mtime_ns) for path in files]
    if initial[0][0] != initial[1][0]:
        raise ValueError("model sizes differ")
    hashes = [digest(path) for path in files]
    if hashes[0] != hashes[1]:
        raise ValueError("model bytes differ; no file replaced")
    if initial != [(path.stat().st_size, path.stat().st_mtime_ns) for path in files]:
        raise ValueError("model file changed during verification")
    linked_before = os.path.samefile(control, candidate)
    if args.apply and not linked_before:
        temp = candidate.with_name(candidate.name + ".link-" + uuid.uuid4().hex)
        try:
            os.link(control, temp)
            if not os.path.samefile(control, temp):
                raise RuntimeError("new link is not the verified model")
            try:
                os.replace(temp, candidate)
            except PermissionError:
                if os.name != "nt" or candidate.stat().st_mode & stat.S_IWRITE:
                    raise
                # Windows refuses replacing an existing read-only destination.
                # Both files have been verified and belong to completed runs;
                # only change permissions, never contents, before replacement.
                original_mode = candidate.stat().st_mode
                os.chmod(candidate, stat.S_IREAD | stat.S_IWRITE)
                try:
                    os.replace(temp, candidate)
                except BaseException:
                    os.chmod(candidate, original_mode)
                    raise
            if not os.path.samefile(control, candidate):
                raise RuntimeError("installed file identity mismatch")
            # Guard both names against accidental future in-place writes.
            os.chmod(control, stat.S_IREAD)
        finally:
            if temp.exists():
                if os.name == "nt":
                    # The temporary link shares the anchor's read-only bit.
                    # Restore the anchor's protection after removing the link.
                    os.chmod(temp, stat.S_IREAD | stat.S_IWRITE)
                temp.unlink()
            os.chmod(control, stat.S_IREAD)
    record = {"control": str(control), "candidate": str(candidate),
              "sha256": hashes[0], "bytes": initial[0][0], "applied": args.apply,
              "already_linked": linked_before, "same_file_after": os.path.samefile(control, candidate),
              "finished_local": time.strftime("%F %T"),
              "scope": "disk storage only; model byte contents and both paths preserved; fresh Fisher not deduplicated"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
