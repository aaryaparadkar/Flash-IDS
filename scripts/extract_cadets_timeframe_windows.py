#!/usr/bin/env python3
"""Extract small chronological CADETS windows from the full original CDM data.

This script is intended for concept-drift experiments where the full CADETS
dataset is too large to train on end-to-end. It scans the whole raw event
stream, then writes compact TSV windows from the beginning, middle, and end of
the original timeline.

Expected raw inputs are the original CADETS files, for example:
  ta1-cadets-e3-official.json
  ta1-cadets-e3-official-2.json

The script also accepts .gz and .tar.gz files.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


EVENT_KEY = "com.bbn.tc.schema.avro.cdm18.Event"
UUID_KEY = "com.bbn.tc.schema.avro.cdm18.UUID"

DEFAULT_INPUTS = [
    "ta1-cadets-e3-official.json",
    "ta1-cadets-e3-official.json.tar.gz",
    "ta1-cadets-e3-official.json.gz",
    "ta1-cadets-e3-official-2.json",
    "ta1-cadets-e3-official-2.json.tar.gz",
    "ta1-cadets-e3-official-2.json.gz",
]


@dataclass(frozen=True)
class WindowSpec:
    name: str
    start_event: int
    end_event: int


def iter_text_lines(path: Path) -> Iterator[str]:
    if path.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(path, "r:gz") as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            for member in members:
                extracted = tf.extractfile(member)
                if extracted is None:
                    continue
                with io.TextIOWrapper(extracted, encoding="utf-8", errors="replace") as f:
                    yield from f
        return

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            yield from f
        return

    with path.open(encoding="utf-8", errors="replace") as f:
        yield from f


def iter_all_lines(paths: Iterable[Path]) -> Iterator[str]:
    for path in paths:
        yield from iter_text_lines(path)


def unwrap_record(line: str) -> tuple[str | None, dict] | None:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None

    datum = obj.get("datum")
    if not isinstance(datum, dict) or not datum:
        return None

    record_type, record = next(iter(datum.items()))
    if not isinstance(record, dict):
        return None
    return record_type, record


def nested_uuid(value) -> str | None:
    if isinstance(value, dict):
        item = value.get(UUID_KEY)
        return item if isinstance(item, str) else None
    return None


def record_uuid(record: dict) -> str | None:
    value = record.get("uuid")
    if isinstance(value, str):
        return value
    return nested_uuid(value)


def record_node_type(record_type: str, record: dict) -> str | None:
    if EVENT_KEY in record_type:
        return None

    type_value = record.get("type")
    if isinstance(type_value, str):
        return type_value

    short_type = record_type.rsplit(".", 1)[-1]
    if short_type in {"MemoryObject", "NetFlowObject", "UnnamedPipeObject"}:
        return short_type
    return None


def first_pass(paths: list[Path]) -> tuple[dict[str, str], int]:
    id_to_type: dict[str, str] = {}
    event_count = 0

    for line in iter_all_lines(paths):
        parsed = unwrap_record(line)
        if parsed is None:
            continue

        record_type, record = parsed
        if EVENT_KEY in record_type:
            event_count += 1
            continue

        uuid = record_uuid(record)
        node_type = record_node_type(record_type, record)
        if uuid and node_type:
            id_to_type[uuid] = node_type

    return id_to_type, event_count


def build_window_specs(total_events: int, window_events: int) -> list[WindowSpec]:
    if total_events <= 0:
        raise ValueError("No CADETS Event records found in the input files.")

    size = min(window_events, max(1, total_events // 5))
    anchors = {
        "reference": 0,
        "early": max(size, total_events // 10),
        "middle": max(0, (total_events // 2) - (size // 2)),
        "late": max(0, total_events - size),
    }

    specs = []
    for name, start in anchors.items():
        end = min(total_events, start + size)
        specs.append(WindowSpec(name=name, start_event=start, end_event=end))
    return specs


def event_to_edges(record: dict, id_to_type: dict[str, str]) -> list[tuple[str, str, str, str, str, int]]:
    src_id = nested_uuid(record.get("subject"))
    if not src_id or src_id not in id_to_type:
        return []

    action = record.get("type")
    timestamp = record.get("timestampNanos")
    if not isinstance(action, str) or timestamp is None:
        return []

    edges = []
    for key in ("predicateObject", "predicateObject2"):
        dst_id = nested_uuid(record.get(key))
        if dst_id and dst_id in id_to_type:
            edges.append(
                (
                    src_id,
                    id_to_type[src_id],
                    dst_id,
                    id_to_type[dst_id],
                    action,
                    int(timestamp),
                )
            )
    return edges


def second_pass(paths: list[Path], specs: list[WindowSpec], id_to_type: dict[str, str], out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        spec.name: (out_dir / f"cadets_full_{spec.name}.tsv").open("w", encoding="utf-8")
        for spec in specs
    }
    counts = {spec.name: 0 for spec in specs}
    event_idx = 0

    try:
        for line in iter_all_lines(paths):
            parsed = unwrap_record(line)
            if parsed is None:
                continue

            record_type, record = parsed
            if EVENT_KEY not in record_type:
                continue

            for spec in specs:
                if spec.start_event <= event_idx < spec.end_event:
                    for edge in event_to_edges(record, id_to_type):
                        handles[spec.name].write(
                            "\t".join(str(value) for value in edge) + "\n"
                        )
                        counts[spec.name] += 1
            event_idx += 1
    finally:
        for handle in handles.values():
            handle.close()

    return counts


def resolve_inputs(raw_inputs: list[str]) -> list[Path]:
    candidates = [Path(p) for p in raw_inputs]
    paths = [p for p in candidates if p.exists()]
    if paths:
        return paths

    defaults = [Path(p) for p in DEFAULT_INPUTS]
    paths = [p for p in defaults if p.exists()]
    if paths:
        return paths

    expected = "\n  ".join(DEFAULT_INPUTS)
    raise FileNotFoundError(
        "No original CADETS files found. Put the original files in the repo root "
        f"or pass them with --input.\nExpected one or more of:\n  {expected}"
    )


def write_metadata(out_dir: Path, paths: list[Path], specs: list[WindowSpec], counts: dict[str, int], total_events: int, node_count: int) -> None:
    metadata = {
        "inputs": [str(p) for p in paths],
        "total_events_scanned": total_events,
        "node_ids_indexed": node_count,
        "windows": [
            {
                "name": spec.name,
                "start_event": spec.start_event,
                "end_event": spec.end_event,
                "event_span": spec.end_event - spec.start_event,
                "output": str(out_dir / f"cadets_full_{spec.name}.tsv"),
                "edge_rows": counts.get(spec.name, 0),
            }
            for spec in specs
        ],
    }
    (out_dir / "cadets_full_timeframe_windows.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Original CADETS JSON/JSON.GZ/TAR.GZ path. Can be used multiple times.",
    )
    parser.add_argument(
        "--window-events",
        type=int,
        default=25000,
        help="Number of original Event records to scan for each extracted timeframe.",
    )
    parser.add_argument(
        "--out-dir",
        default="cadets_full_timeframes",
        help="Directory for extracted TSV windows.",
    )
    args = parser.parse_args()

    paths = resolve_inputs(args.input)
    out_dir = Path(args.out_dir)

    print("Scanning full CADETS files to index node types and count events...")
    id_to_type, total_events = first_pass(paths)
    print(f"Indexed {len(id_to_type):,} node ids; counted {total_events:,} events.")

    specs = build_window_specs(total_events, args.window_events)
    for spec in specs:
        print(f"{spec.name}: events {spec.start_event:,} to {spec.end_event:,}")

    print("Extracting timeframe edge TSVs...")
    counts = second_pass(paths, specs, id_to_type, out_dir)
    write_metadata(out_dir, paths, specs, counts, total_events, len(id_to_type))

    for spec in specs:
        print(f"{out_dir / f'cadets_full_{spec.name}.tsv'}: {counts[spec.name]:,} edges")
    print(f"Metadata: {out_dir / 'cadets_full_timeframe_windows.json'}")


if __name__ == "__main__":
    main()
