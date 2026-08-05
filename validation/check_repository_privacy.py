"""Fail when tracked files or Git history contain data-shaped artifacts."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLOCKED_SUFFIXES = {
    ".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".sas7bdat", ".dta",
    ".sav", ".feather", ".xpt", ".db", ".sqlite", ".sqlite3", ".pkl",
    ".pickle", ".pt", ".ckpt", ".npy", ".npz", ".docx", ".pptx",
    ".pdf", ".log", ".jsonl", ".ndjson", ".zip", ".7z", ".tar",
    ".tgz", ".pem", ".pfx", ".p12",
}
BLOCKED_SEGMENTS = {"empirical", "raw", "data", "derived", "derived_phi", "outputs", "runs"}
BLOCKED_BASENAMES = {"rdoc_llm_scorer.csv", ".env", ".openai_key"}
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "OpenAI-style API key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "US Social Security number": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True, encoding="utf-8", errors="replace"
    )


def _blocked_path(rel: str) -> str | None:
    path = Path(rel)
    parts = {part.lower() for part in path.parts[:-1]}
    lower_name = path.name.lower()
    if path.suffix.lower() in BLOCKED_SUFFIXES:
        return "data or document file extension"
    if lower_name.endswith((".tar.gz", ".csv.gz")):
        return "compressed data or archive filename"
    if parts.intersection(BLOCKED_SEGMENTS):
        return "protected or generated data directory"
    if lower_name in BLOCKED_BASENAMES or lower_name.startswith("credentials"):
        return "known protected or credential filename"
    if re.fullmatch(r"amd\d+\.sas7bdat", lower_name):
        return "raw EHR SAS filename"
    return None


def main() -> None:
    failures: list[str] = []
    tracked = [
        line for line in _git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
        if line
    ]
    historical = {
        line for line in _git("log", "--all", "--name-only", "--pretty=format:").splitlines() if line
    }
    for scope, names in (("tracked", tracked), ("history", sorted(historical))):
        for name in names:
            reason = _blocked_path(name)
            if reason:
                failures.append(f"{scope}: {name} ({reason})")

    # Scan only current text source. The script reports the file and pattern name,
    # never the matching content.
    for name in tracked:
        path = ROOT / name
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"tracked: {name} (unexpected binary content)")
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"tracked: {name} ({label})")

    # Scan every reasonably sized text blob reachable from any ref, including
    # content removed from the current branch. Never print matching content.
    seen_blobs: set[str] = set()
    for object_line in _git("rev-list", "--objects", "--all").splitlines():
        oid = object_line.split(" ", 1)[0]
        if oid in seen_blobs:
            continue
        if _git("cat-file", "-t", oid).strip() != "blob":
            continue
        seen_blobs.add(oid)
        size = int(_git("cat-file", "-s", oid).strip())
        if size > 2_000_000:
            failures.append(f"history object {oid[:12]} (unexpected blob larger than 2 MB)")
            continue
        blob = subprocess.check_output(["git", "-C", str(ROOT), "cat-file", "blob", oid])
        try:
            text = blob.decode("utf-8")
        except UnicodeDecodeError:
            failures.append(f"history object {oid[:12]} (unexpected binary content)")
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"history object {oid[:12]} ({label})")

    if failures:
        print("Repository privacy guard FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print(
        f"Repository privacy guard passed for {len(tracked)} tracked paths and "
        f"{len(historical)} historical paths and {len(seen_blobs)} historical blobs."
    )


if __name__ == "__main__":
    main()
