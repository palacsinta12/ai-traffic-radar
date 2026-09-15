"""Repository verification smoke checks for the radar BEV project."""

import argparse
import logging
import sys
from pathlib import Path

from config import PROJECT_DIR, VERIFICATION_PATTERNS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Check repository polish and required directories.")
    parser.add_argument("--root", default=str(PROJECT_DIR), help="Repository root path.")
    return parser.parse_args()


def verify_repo(root: str | Path):
    root = Path(root)
    checks = []

    required_dirs = [
        root / "scripts",
        root / "measurements",
        root / "dataset",
        root / "runs",
        root / "assets",
    ]

    for directory in required_dirs:
        if directory.exists():
            checks.append(f"Directory found: {directory}")
        else:
            checks.append(f"Missing directory: {directory}")

    script_files = sorted((root / "scripts").glob("*.py"))
    module_scan_issues = []
    for file in script_files:
        if file.name == "config.py" or file.name == "verify_repo.py":
            continue

        try:
            text = file.read_text(encoding="utf-8")
        except Exception:
            continue

        for pattern in VERIFICATION_PATTERNS:
            if pattern.lower() in text.lower():
                module_scan_issues.append(f"{file.name}: contains {pattern}")

    if module_scan_issues:
        logging.warning("Repository wording scan found: %s", module_scan_issues)
    else:
        logging.info("Repository wording scan complete: no markers found.")

    for check in checks:
        logging.info(check)

    return len(module_scan_issues) == 0


if __name__ == "__main__":
    args = parse_args()
    ok = verify_repo(args.root)
    sys.exit(0 if ok else 1)
