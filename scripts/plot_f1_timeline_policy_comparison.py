#!/usr/bin/env python3
"""Generate CADETS drift timeline graphs.

Outputs:
  1. F1 vs timeline
  2. PSI(action) vs timeline
  3. F1 vs PSI(action)
  4. Ground-truth coverage vs timeline
"""

from __future__ import annotations

import argparse
import csv
import json
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
DEFAULT_WINDOW_DIR = "drift_windows"
STRATEGY_RENAME = {
    "Naive drift policy": "Drift policy",
}
EXCLUDED_STRATEGIES = {"Drift policy"}


def window_index(window: str) -> int:
    return ORDER.index(window)


def read_points(path: Path) -> dict[str, list[tuple[str, str, float]]]:
    grouped: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            window = row["window"]
            strategy = row["strategy"]
            if window not in ORDER or strategy in EXCLUDED_STRATEGIES:
                continue
            grouped[STRATEGY_RENAME.get(strategy, strategy)].append((window, row["timeline"], float(row["f1"])))

    for strategy, points in grouped.items():
        points.sort(key=lambda p: window_index(p[0]))
        if len(points) != 3:
            raise ValueError(f"{strategy} does not have early, middle, and late points.")
    return grouped


def read_timeline_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            strategy = row.get("strategy", "")
            if row.get("window") not in ORDER or strategy in EXCLUDED_STRATEGIES:
                continue
            row = dict(row)
            row["strategy"] = STRATEGY_RENAME.get(strategy, strategy)
            rows.append(row)
    if not rows:
        raise ValueError(f"No timeline rows found in {path}")
    return rows


def read_series(rows: list[dict[str, str]], value_key: str) -> dict[str, list[tuple[str, float]]]:
    series: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in rows:
        series[row["strategy"]].append((row["window"], float(row[value_key])))
    for values in series.values():
        values.sort(key=lambda item: window_index(item[0]))
    return series


def window_paths_from_dir(window_dir: Path) -> dict[str, Path]:
    candidates = {
        "early": [window_dir / "early.tsv", window_dir / "window_1.tsv"],
        "middle": [window_dir / "middle.tsv", window_dir / "window_2.tsv"],
        "late": [window_dir / "late.tsv", window_dir / "window_3.tsv"],
    }
    resolved = {}
    for window, paths in candidates.items():
        for path in paths:
            if path.exists():
                resolved[window] = path
                break
        if window not in resolved:
            raise FileNotFoundError(f"No TSV found for {window} in {window_dir}")
    return resolved


def read_gt_coverage(gt_path: Path, window_paths: dict[str, Path]) -> list[tuple[str, int, int]]:
    gt_ids = set(json.loads(gt_path.read_text(encoding="utf-8")))
    coverage = []
    for window in ORDER:
        ids = set()
        with window_paths[window].open(encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    ids.add(parts[0])
                    ids.add(parts[2])
        coverage.append((window, len(ids & gt_ids), len(gt_ids)))
    return coverage


def text(x: float, y: float, value: str, *, size: int = 14, anchor: str = "middle", weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-size="{size}" font-weight="{weight}" fill="#1f2933">{value}</text>'
    )


def svg_start(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif}.axis{stroke:#4b5965;stroke-width:1.4}.grid{stroke:#d9e1e8;stroke-width:1}.line{fill:none;stroke-width:3.2;stroke-linecap:round;stroke-linejoin:round}.threshold{stroke:#8b1e3f;stroke-width:2;stroke-dasharray:7 6}</style>',
        text(width / 2, 36, title, size=24, weight="700"),
        text(width / 2, 64, subtitle, size=14),
    ]


def render_axes(
    parts: list[str],
    width: int,
    height: int,
    left: int,
    right: int,
    top: int,
    bottom: int,
    y_max: float,
    y_label: str,
    x_label: str,
    *,
    tick_count: int = 5,
) -> tuple[float, float]:
    plot_w = width - left - right
    plot_h = height - top - bottom

    def y_pos(value: float) -> float:
        return top + plot_h - (value / y_max) * plot_h

    for i in range(tick_count + 1):
        tick = y_max * i / tick_count
        y = y_pos(tick)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}"/>')
        parts.append(text(left - 14, y + 5, f"{tick:.3f}", size=12, anchor="end"))

    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
            f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}"/>',
            text(28, top + plot_h / 2, y_label, size=15, weight="700").replace(
                "<text ", '<text transform="rotate(-90 28 {0:.1f})" '.format(top + plot_h / 2)
            ),
            text(left + plot_w / 2, height - 28, x_label, size=15, weight="700"),
        ]
    )
    return plot_w, plot_h


def render_timeline_x_labels(parts: list[str], x_pos, top: int, plot_h: float) -> None:
    for index, window in enumerate(ORDER):
        x = x_pos(index)
        parts.append(f'<line class="axis" x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 7}"/>')
        parts.append(text(x, top + plot_h + 30, LABELS[window], size=14, weight="700"))
        parts.append(text(x, top + plot_h + 50, window, size=12))


def render_legend(parts: list[str], items: list[str], x: float, y: float) -> None:
    parts.append(text(x, y - 16, "Series", size=15, anchor="start", weight="700"))
    for i, item in enumerate(items):
        yy = y + i * 34
        color = COLORS.get(item, "#5f6368")
        parts.append(f'<line x1="{x}" y1="{yy}" x2="{x + 28}" y2="{yy}" stroke="{color}" stroke-width="4" stroke-linecap="round"/>')
        parts.append(text(x + 40, yy + 5, item, size=14, anchor="start"))


def render_f1_timeline(points: dict[str, list[tuple[str, str, float]]], output: Path) -> None:
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

    parts = svg_start(width, height, "Graph 1: F1 Score vs Timeline", "Shows model performance over time.")
    render_axes(parts, width, height, left, right, top, bottom, y_max, "F1 score", "Timeline window")
    render_timeline_x_labels(parts, x_pos, top, plot_h)

    for strategy, series in points.items():
        color = COLORS.get(strategy, "#5f6368")
        polyline = " ".join(f"{x_pos(i):.1f},{y_pos(value):.1f}" for i, (_, _, value) in enumerate(series))
        parts.append(f'<polyline class="line" points="{polyline}" stroke="{color}"/>')
        for i, (_, _, value) in enumerate(series):
            x = x_pos(i)
            y = y_pos(value)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.8" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
            parts.append(text(x, y - 13, f"{value:.4f}", size=12, weight="700"))

    render_legend(parts, list(points), width - right + 42, top + 24)

    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def render_psi_timeline(psi_series: dict[str, list[tuple[str, float]]], output: Path) -> None:
    width, height = 980, 580
    left, right, top, bottom = 92, 210, 90, 112
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_max = max(0.75, max(value for series in psi_series.values() for _, value in series) + 0.08)

    def x_pos(index: int) -> float:
        return left + index * (plot_w / 2)

    def y_pos(value: float) -> float:
        return top + plot_h - (value / y_max) * plot_h

    parts = svg_start(width, height, "Graph 2: PSI(action) vs Timeline", "Shows concept drift over time. Threshold line is 0.2.")
    render_axes(parts, width, height, left, right, top, bottom, y_max, "PSI(action)", "Timeline window")
    render_timeline_x_labels(parts, x_pos, top, plot_h)
    threshold_y = y_pos(0.2)
    parts.append(f'<line class="threshold" x1="{left}" y1="{threshold_y:.1f}" x2="{width - right}" y2="{threshold_y:.1f}"/>')
    parts.append(text(width - right - 4, threshold_y - 8, "drift threshold = 0.2", size=12, anchor="end", weight="700"))

    for strategy, series in psi_series.items():
        color = COLORS.get(strategy, "#5f6368")
        polyline = " ".join(f"{x_pos(i):.1f},{y_pos(value):.1f}" for i, (_, value) in enumerate(series))
        parts.append(f'<polyline class="line" points="{polyline}" stroke="{color}"/>')
        for i, (_, value) in enumerate(series):
            x = x_pos(i)
            y = y_pos(value)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.8" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
            parts.append(text(x, y - 13, f"{value:.3f}", size=12, weight="700"))

    render_legend(parts, list(psi_series), width - right + 42, top + 24)
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def render_f1_vs_psi(rows: list[dict[str, str]], output: Path) -> None:
    width, height = 980, 580
    left, right, top, bottom = 92, 210, 90, 96
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_psi = max(float(row["psi_action"]) for row in rows)
    max_f1 = max(float(row["f1"]) for row in rows)
    x_max = max(0.75, max_psi + 0.08)
    y_max = max(0.04, max_f1 + 0.02)

    def x_pos(value: float) -> float:
        return left + (value / x_max) * plot_w

    def y_pos(value: float) -> float:
        return top + plot_h - (value / y_max) * plot_h

    parts = svg_start(width, height, "Graph 3: F1 and PSI(action) Together", "Shows whether detected drift aligns with performance change.")
    render_axes(parts, width, height, left, right, top, bottom, y_max, "F1 score", "PSI(action)")
    for i in range(6):
        tick = x_max * i / 5
        x = x_pos(tick)
        parts.append(f'<line class="axis" x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 7}"/>')
        parts.append(text(x, top + plot_h + 28, f"{tick:.2f}", size=12))

    threshold_x = x_pos(0.2)
    parts.append(f'<line class="threshold" x1="{threshold_x:.1f}" y1="{top}" x2="{threshold_x:.1f}" y2="{top + plot_h}"/>')
    parts.append(text(threshold_x + 8, top + 16, "0.2 threshold", size=12, anchor="start", weight="700"))

    for row in rows:
        color = COLORS.get(row["strategy"], "#5f6368")
        x = x_pos(float(row["psi_action"]))
        y = y_pos(float(row["f1"]))
        label = row["window"]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.2" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
        parts.append(text(x + 10, y - 8, label, size=12, anchor="start", weight="700"))

    render_legend(parts, list(dict.fromkeys(row["strategy"] for row in rows)), width - right + 42, top + 24)
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def render_gt_coverage(coverage: list[tuple[str, int, int]], output: Path) -> None:
    width, height = 900, 560
    left, right, top, bottom = 92, 54, 90, 112
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_max = max(total for _, _, total in coverage)
    bar_w = plot_w / 7

    def x_center(index: int) -> float:
        return left + index * (plot_w / 2)

    def y_pos(value: float) -> float:
        return top + plot_h - (value / y_max) * plot_h

    parts = svg_start(width, height, "Graph 4: Ground-Truth Coverage", "Explains how reliable each F1 point is.")
    render_axes(parts, width, height, left, right, top, bottom, y_max, "GT IDs in window", "Timeline window")
    render_timeline_x_labels(parts, x_center, top, plot_h)
    parts.append(text(width - right, top - 10, f"Global GT IDs = {y_max}", size=12, anchor="end", weight="700"))

    for index, (window, overlap, total) in enumerate(coverage):
        x = x_center(index) - bar_w / 2
        y = y_pos(overlap)
        h = top + plot_h - y
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#3B6EA8"/>')
        parts.append(text(x + bar_w / 2, y - 10, f"{overlap}/{total}", size=13, weight="700"))
        parts.append(text(x + bar_w / 2, y + h + 72, f"ceiling {overlap / total:.3f}", size=12))

    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/cadets_full_f1_timeline_with_drift_policy.csv")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--gt", default="data_files/cadets.json")
    parser.add_argument("--window-dir", default=DEFAULT_WINDOW_DIR)
    args = parser.parse_args()

    rows = read_timeline_rows(Path(args.input))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        out_dir / "graph1_f1_vs_timeline.svg",
        out_dir / "graph2_psi_action_vs_timeline.svg",
        out_dir / "graph3_f1_vs_psi_action.svg",
        out_dir / "graph4_ground_truth_coverage.svg",
    ]
    render_f1_timeline(read_points(Path(args.input)), outputs[0])
    render_psi_timeline(read_series(rows, "psi_action"), outputs[1])
    render_f1_vs_psi(rows, outputs[2])
    render_gt_coverage(read_gt_coverage(Path(args.gt), window_paths_from_dir(Path(args.window_dir))), outputs[3])

    # Keep the original filename updated for existing report links.
    legacy_output = out_dir / "cadets_full_f1_timeline_with_drift_policy.svg"
    render_f1_timeline(read_points(Path(args.input)), legacy_output)

    for output in outputs + [legacy_output]:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
