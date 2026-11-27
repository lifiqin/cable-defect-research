"""Fail closed if a staged change is outside the reviewed source allowlist."""

import re
import subprocess

ALLOWED = re.compile(r"(?:\.gitignore|README\.md|pyproject\.toml|check_staged\.py|"
                     r"\.github/workflows/ci\.yml|docs/related_work\.md|"
                     r"cable_research/[a-z_]+\.py|tests/test_[a-z_]+\.py)")
SENSITIVE = re.compile(r"(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
                       r"sk-[A-Za-z0-9]{20,}|IMG_\d{4,}|[A-Za-z]:[/\\]Users[/\\])")


def main():
    raw = subprocess.check_output(["git", "diff", "--cached", "--name-only", "-z"])
    names = [n.decode("utf-8") for n in raw.split(b"\0") if n]
    if not names:
        raise ValueError("nothing staged")
    for name in names:
        if not ALLOWED.fullmatch(name):
            raise ValueError(f"not on publication allowlist: {name}")
        data = subprocess.check_output(["git", "show", f":{name}"])
        if len(data) > 256_000 or b"\0" in data:
            raise ValueError(f"unexpected binary or large file: {name}")
        if SENSITIVE.search(data.decode("utf-8")):
            raise ValueError(f"potential private data or credential in: {name}")
    print(f"Publication allowlist passed for {len(names)} staged source files")


if __name__ == "__main__":
    main()
