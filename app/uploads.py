"""Upload validation helpers: content sniffing and filename sanitization.

Defends against the cases called out in TECHNICAL_SPEC §7.3: don't trust the
declared extension/MIME type, and never derive storage paths from client
input (paths are server-generated in `app.storage`).
"""

from __future__ import annotations

import re

# Magic-number prefixes for common binary formats. A genuine CSV will never
# start with these, regardless of what Content-Type the client claims.
_BINARY_MAGIC_PREFIXES: tuple[bytes, ...] = (
    b"PK",  # zip / xlsx / docx / ...
    b"\x89PNG",
    b"\xff\xd8\xff",  # jpeg
    b"GIF8",
    b"%PDF",
    b"\x1f\x8b",  # gzip
    b"BM",  # bmp
    b"\x7fELF",  # elf binary
    b"MZ",  # windows pe
)

_FILENAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._ -]+")


def sanitize_filename(filename: str, fallback: str = "dataset.csv", max_length: int = 200) -> str:
    """Strip path components and disallowed characters from a client-supplied
    filename. Result is used only as a display `name`, never as a storage path.
    """
    name = (filename or "").strip().replace("\\", "/").split("/")[-1]
    name = _FILENAME_SANITIZE_RE.sub("_", name)
    name = name.strip("._ ") or fallback
    return name[:max_length]


def looks_like_csv(data: bytes, sample_size: int = 4096) -> bool:
    """Best-effort content sniff: reject obvious binary formats and content
    that isn't decodable as text.
    """
    sample = data[:sample_size]
    if not sample:
        return False
    for magic in _BINARY_MAGIC_PREFIXES:
        if sample.startswith(magic):
            return False
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        try:
            sample.decode("latin-1")
        except UnicodeDecodeError:
            return False
    return True
