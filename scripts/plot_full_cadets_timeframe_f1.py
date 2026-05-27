#!/usr/bin/env python3
"""Plot F1 score across full-CADETS timeframe windows.

Input CSV format can be either a simple three-row file:
  timeline,window,f1
  Early sample,early,0.123
  Middle sample,middle,0.234
  Late sample,late,0.190

or the project's rolling_drift_summary.csv with test_window,strategy,f1 columns.
Optional columns such as precision, recall, strategy, or notes are ignored.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ORDER = ["early", "middle", "late"]
DEFAULT_LABELS = {
    "early": "Early sample",
    "middle": "Middle sample",
    "late": "Late sample",
}
ROLLING_WINDOW_MAP = {
    "W1": "early",
    "W2": "middle",
    "W3": "late",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    if "f1" not in rows[0]:
        raise ValueError("Input CSV must contain an f1 column.")
    return rows


def normalise_rows(rows: list[dict[str, str]]) -> list[tuple[str, str, float]]:
    if {"test_window", "strategy", "f1"}.issubset(rows[0]):
        best_by_window: dict[str, tuple[str, float]] = {}
        for row in rows:
            window = ROLLING_WINDOW_MAP.get(row["test_window"])
            if not window:
                continue
            f1 = float(row["f1"])
            current = best_by_window.get(window)
            if current is None or f1 > current[1]:
                best_by_window[window] = (row["strategy"], f1)
        points = [
            (window, f"{DEFAULT_LABELS[window]} ({best_by_window[window][0]})", best_by_window[window][1])
            for window in ORDER
            if window in best_by_window
        ]
        if len(points) != 3:
            raise ValueError("Expected rolling results for W1, W2, and W3.")
        return points

    points = []
    for row in rows:
        window = (row.get("window") or row.get("timeframe") or "").strip().lower()
        timeline = (row.get("timeline") or DEFAULT_LABELS.get(window) or window.title()).strip()
        if window not in ORDER:
            continue
        points.append((window, timeline, float(row["f1"])))
    points.sort(key=lambda item: ORDER.index(item[0]))
    if len(points) != 3:
        raise ValueError("Expected one row for each window: early, middle, late.")
    return points


def text(x: float, y: float, value: str, *, size: int = 14, anchor: str = "middle", weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-size="{size}" font-weight="{weight}" fill="#1f2933">{value}</text>'
    )


def render_svg(points: list[tuple[str, str, float]], output: Path) -> None:
    width, height = 920, 560
    left, right, top, bottom = 92, 54, 88, 104
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_f1 = max(value for _, _, value in points)
    y_max = max(0.1, min(1.0, ((max_f1 + 0.1) * 10 // 1 + 1) / 10))

    def x_pos(index: int) -> float:
        return left + index * (plot_w / (len(points) - 1))

    def y_pos(value: float) -> float:
        return top + plot_h - (value / y_max) * plot_h

    color = "#2F6F73"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif}.axis{stroke:#4b5965;stroke-width:1.4}.grid{stroke:#d9e1e8;stroke-width:1}.line{fill:none;stroke-width:3.4;stroke-linecap:round;stroke-linejoin:round}</style>',
        text(width / 2, 36, "F1 Score vs CADETS Timeline", size=24, weight="700"),
        text(width / 2, 64, "Small early, middle, and late slices are extracted from the full original CADETS event stream.", size=14),
    ]

    tick_count = 5
    for i in range(tick_count + 1):
        tick = y_max * i / tick_count
        y = y_pos(tick)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}"/>')
        parts.append(text(left - 14, y + 5, f"{tick:.2f}", size=12, anchor="end"))

    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
            f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}"/>',
            text(28, top + plot_h / 2, "F1 score", size=15, weight="700").replace(
                "<text ", '<text transform="rotate(-90 28 {0:.1f})" '.format(top + plot_h / 2)
            ),
            text(left + plot_w / 2, height - 28, "Timeline from original CADETS dataset", size=15, weight="700"),
        ]
    )

    polyline = []
    for index, (window, label, f1) in enumerate(points):
        x = x_pos(index)
        y = y_pos(f1)
        polyline.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<line class="axis" x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 7}"/>')
        parts.append(text(x, top + plot_h + 30, label, size=14, weight="700"))
        parts.append(text(x, top + plot_h + 50, window, size=12))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
        parts.append(text(x, y - 14, f"{f1:.3f}", size=13, weight="700"))

    parts.insert(-12, f'<polyline class="line" points="{" ".join(polyline)}" stroke="{color}"/>')
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/cadets_full_timeframe_f1.csv")
    parser.add_argument("--output", default="results/cadets_full_timeframe_f1.svg")
    args = parser.parse_args()

    points = normalise_rows(read_rows(Path(args.input)))
    render_svg(points, Path(args.output))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
