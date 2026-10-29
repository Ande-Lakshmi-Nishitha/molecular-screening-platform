"""Small helpers shared by the training scripts (MSP-22)."""
import datetime
import hashlib
import json
from pathlib import Path


def file_sha256(path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, read in chunks, as lowercase hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now_iso() -> str:
    """Current UTC time, for example 2026-10-29T04:30:00Z."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path, data) -> None:
    """Write readable, key-sorted JSON with a trailing newline (creates folders)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
