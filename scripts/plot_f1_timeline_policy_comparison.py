#!/usr/bin/env python3
"""Plot static F1 vs drift-policy F1 across CADETS timeline windows."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ORDER = ["early", "middle", "late"]
LABELS = {
    "early": "Early sample",
    "middle": "Middle sample",
    "late": "Late sample",
}
COLORS = {
    "Static baseline": "#2F6F73",
    "Drift policy": "#C7503E",
}


def read_points(path: Path) -> dict[str, list[tuple[str, str, float]]]:
    grouped: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            window = row["window"]
            if window not in ORDER:
                continue
            grouped[row["strategy"]].append((window, row["timeline"], float(row["f1"])))

    for strategy, points in grouped.items():
        points.sort(key=lambda p: ORDER.index(p[0]))
        if len(points) != 3:
            raise ValueError(f"{strategy} does not have early, middle, and late points.")
    return grouped


def text(x: float, y: float, value: str, *, size: int = 14, anchor: str = "middle", weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-size="{size}" font-weight="{weight}" fill="#1f2933">{value}</text>'
    )


def render_svg(points: dict[str, list[tuple[str, str, float]]], output: Path) -> None:
    width, height = 980, 580
    left, right, top, bottom = 92, 210, 90, 112
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_f1 = max(value for series in points.values() for _, _, value in series)
    y_max = max(0.04, min(1.0, max_f1 + 0.02))

    def x_pos(index: int) -> float:
        return left + index * (plot_w / 2)

    def y_pos(value: float) -> float:
        return top + plot_h - (value / y_max) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif}.axis{stroke:#4b5965;stroke-width:1.4}.grid{stroke:#d9e1e8;stroke-width:1}.line{fill:none;stroke-width:3.2;stroke-linecap:round;stroke-linejoin:round}</style>',
        text(width / 2, 36, "F1 Score vs CADETS Timeline", size=24, weight="700"),
        text(width / 2, 64, "Static baseline compared with the applied drift policy.", size=14),
    ]

    tick_count = 5
    for i in range(tick_count + 1):
        tick = y_max * i / tick_count
        y = y_pos(tick)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}"/>')
        parts.append(text(left - 14, y + 5, f"{tick:.3f}", size=12, anchor="end"))

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

    for index, window in enumerate(ORDER):
        x = x_pos(index)
        parts.append(f'<line class="axis" x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 7}"/>')
        parts.append(text(x, top + plot_h + 30, LABELS[window], size=14, weight="700"))
        parts.append(text(x, top + plot_h + 50, window, size=12))

    for strategy, series in points.items():
        color = COLORS.get(strategy, "#5f6368")
        polyline = " ".join(f"{x_pos(i):.1f},{y_pos(value):.1f}" for i, (_, _, value) in enumerate(series))
        parts.append(f'<polyline class="line" points="{polyline}" stroke="{color}"/>')
        for i, (_, _, value) in enumerate(series):
            x = x_pos(i)
            y = y_pos(value)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.8" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
            parts.append(text(x, y - 13, f"{value:.4f}", size=12, weight="700"))

    legend_x = width - right + 42
    legend_y = top + 24
    parts.append(text(legend_x, legend_y - 16, "Series", size=15, anchor="start", weight="700"))
    for i, strategy in enumerate(points):
        y = legend_y + i * 34
        color = COLORS.get(strategy, "#5f6368")
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="4" stroke-linecap="round"/>')
        parts.append(text(legend_x + 40, y + 5, strategy, size=14, anchor="start"))

    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/cadets_full_f1_timeline_with_drift_policy.csv")
    parser.add_argument("--output", default="results/cadets_full_f1_timeline_with_drift_policy.svg")
    args = parser.parse_args()

    render_svg(read_points(Path(args.input)), Path(args.output))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
