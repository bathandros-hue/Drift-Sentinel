"""Compare CSV datasets without any third-party or hosted services."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import analyze, load_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect tabular distribution and schema drift")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--psi-threshold", type=float, default=0.2)
    parser.add_argument("--missing-threshold", type=float, default=0.1)
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args()
    report = analyze(
        load_csv(args.baseline), load_csv(args.current),
        args.psi_threshold, args.missing_threshold,
    )
    print(json.dumps(report, indent=2))
    if args.fail_on_drift and report["flagged_columns"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
