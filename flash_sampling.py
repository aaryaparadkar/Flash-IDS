#!/usr/bin/env python3
"""Research-grade CADETS/CDM JSONL splitting utilities.

The important rule: split by event time before any optional downsampling.
Random line sampling can leak future context into training and can destroy graph
structure by keeping events while dropping their entity records.
"""

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


EVENT_KEY = "com.bbn.tc.schema.avro.cdm18.Event"
UUID_KEY = "com.bbn.tc.schema.avro.cdm18.UUID"


@dataclass
class EventRecord:
    timestamp: int
    line_no: int
    raw: str
    refs: Tuple[str, ...]


def _datum(record):
    return record.get("datum", {}) if isinstance(record, dict) else {}


def _record_type(record):
    datum = _datum(record)
    if not datum:
        return "unknown"
    key = next(iter(datum.keys()))
    return key.rsplit(".", 1)[-1]


def _extract_entity_uuid(record) -> Optional[str]:
    datum = _datum(record)
    if not datum:
        return None
    payload = next(iter(datum.values()))
    if isinstance(payload, dict):
        uuid = payload.get("uuid")
        if isinstance(uuid, str):
            return uuid
    return None


def _extract_event(record, line_no: int, raw: str) -> Optional[EventRecord]:
    datum = _datum(record)
    if EVENT_KEY not in datum:
        return None
    event = datum[EVENT_KEY]
    refs = []
    for field in ("subject", "predicateObject", "predicateObject2"):
        value = event.get(field)
        if isinstance(value, dict):
            ref = value.get(UUID_KEY)
            if ref and ref != "null":
                refs.append(ref)
    return EventRecord(
        timestamp=int(event.get("timestampNanos", 0) or 0),
        line_no=line_no,
        raw=raw,
        refs=tuple(refs),
    )


def _keep_event(event: EventRecord, sample_rate: float, seed: int) -> bool:
    if sample_rate >= 1.0:
        return True
    key = f"{seed}:{event.timestamp}:{event.line_no}:{','.join(event.refs)}".encode()
    bucket = int(hashlib.sha256(key).hexdigest()[:16], 16) / float(16 ** 16)
    return bucket < sample_rate


def _split_events(events: List[EventRecord], train_ratio: float, val_ratio: float):
    events = sorted(events, key=lambda e: (e.timestamp, e.line_no))
    n = len(events)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return {
        "train": events[:train_end],
        "val": events[train_end:val_end],
        "test": events[val_end:],
    }


def split_cdm_jsonl(
    input_path: str,
    out_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    sample_rate: float = 1.0,
    seed: int = 42,
) -> Dict:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must satisfy train_ratio > 0, val_ratio >= 0, train + val < 1")
    if sample_rate <= 0 or sample_rate > 1:
        raise ValueError("sample_rate must be in (0, 1]")

    os.makedirs(out_dir, exist_ok=True)
    events: List[EventRecord] = []
    entities: Dict[str, str] = {}
    type_counts: Dict[str, int] = {}
    total_lines = 0

    with open(input_path, "r") as handle:
        for line_no, raw in enumerate(handle):
            total_lines += 1
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            record_type = _record_type(record)
            type_counts[record_type] = type_counts.get(record_type, 0) + 1
            event = _extract_event(record, line_no, raw)
            if event is not None:
                events.append(event)
                continue
            uuid = _extract_entity_uuid(record)
            if uuid:
                entities[uuid] = raw

    split_events = _split_events(events, train_ratio, val_ratio)
    stats = {
        "input_file": os.path.abspath(input_path),
        "total_lines_scanned": total_lines,
        "record_type_counts": type_counts,
        "split_method": "chronological_event_time_then_optional_deterministic_downsample",
        "sample_rate": sample_rate,
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": 1.0 - train_ratio - val_ratio,
        },
        "splits": {},
    }

    for split_name, split in split_events.items():
        sampled = [event for event in split if _keep_event(event, sample_rate, seed)]
        refs = {ref for event in sampled for ref in event.refs}
        entity_lines = [entities[ref] for ref in sorted(refs) if ref in entities]
        out_path = os.path.join(out_dir, f"{split_name}.jsonl")
        with open(out_path, "w") as out:
            out.writelines(entity_lines)
            out.writelines(event.raw for event in sampled)
        timestamps = [event.timestamp for event in sampled]
        stats["splits"][split_name] = {
            "path": os.path.abspath(out_path),
            "events_before_sampling": len(split),
            "events_after_sampling": len(sampled),
            "entity_records_written": len(entity_lines),
            "timestamp_min": min(timestamps) if timestamps else None,
            "timestamp_max": max(timestamps) if timestamps else None,
        }

    stats_path = os.path.join(out_dir, "sample_stats.json")
    with open(stats_path, "w") as out:
        json.dump(stats, out, indent=2)
    return stats


def main(argv: Optional[Iterable[str]] = None):
    parser = argparse.ArgumentParser(description="Chronological CDM JSONL split builder")
    parser.add_argument("--input", required=True, help="Raw CDM JSONL file")
    parser.add_argument("--out-dir", default="cadets_sampled_fair", help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    stats = split_cdm_jsonl(
        input_path=args.input,
        out_dir=args.out_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        sample_rate=args.sample_rate,
        seed=args.seed,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
