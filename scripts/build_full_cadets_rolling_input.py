#!/usr/bin/env python3
"""Build a four-window rolling-drift TSV from extracted full-CADETS slices."""

from __future__ import annotations

import argparse
from pathlib import Path


WINDOWS = ["reference", "early", "middle", "late"]


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        return [line for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="cadets_full_timeframes")
    parser.add_argument("--output", default="cadets_full_timeframes/cadets_full_rolling_input.tsv")
    parser.add_argument(
        "--rows-per-window",
        type=int,
        default=0,
        help="Rows to keep per window. Default uses the smallest available extracted window.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    window_lines = {
        name: read_lines(input_dir / f"cadets_full_{name}.tsv")
        for name in WINDOWS
    }

    min_rows = min(len(lines) for lines in window_lines.values())
    if min_rows == 0:
        raise ValueError("At least one extracted window has zero TSV rows.")

    rows_per_window = args.rows_per_window or min_rows
    if rows_per_window > min_rows:
        raise ValueError(
            f"--rows-per-window={rows_per_window} is larger than the smallest window ({min_rows})."
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for name in WINDOWS:
            f.writelines(window_lines[name][:rows_per_window])

    print(f"Wrote {output}")
    print(f"Rows per window: {rows_per_window:,}")
    print(f"Total rows: {rows_per_window * len(WINDOWS):,}")
    for index, name in enumerate(WINDOWS):
        print(f"W{index}: {name}")


if __name__ == "__main__":
    main()
