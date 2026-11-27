#!/usr/bin/env python3
"""Flatten the Sec 2-3 harness outputs into one CSV that verify can read.

    flatten_outputs.py <harness/out> <results/sec3_values.csv>

* every scalar leaf of every out/*.json becomes a row (source, key, value); list elements are keyed by index,
  or by their 'd' / 'arm' / 'k' field when they have one;
* out/bitsvsdims.txt (Table 2) becomes rows keyed 'bitsvsdims.<budget>.<arm>.<column>'.
Values are copied as written by the harness, never recomputed.
"""
import csv, json, re, sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = []

def walk(x, key, source):
    if isinstance(x, dict):
        for k, v in x.items():
            walk(v, f"{key}.{k}" if key else str(k), source)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            tag = next((f"{f}={v[f]}" for f in ("d", "arm", "k", "name") if isinstance(v, dict) and f in v), str(i))
            walk(v, f"{key}[{tag}]", source)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        rows.append((source, key, repr(x)))

for f in sorted(src.glob("*.json")):
    try:
        walk(json.load(open(f)), "", f.name)
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass

budget = None
for line in open(src / "bitsvsdims.txt"):
    m = re.match(r"\s*(\d+) B/vector", line)
    if m:
        budget = m.group(1); continue
    m = re.match(r"\s+(\S+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s*$", line)
    if m and budget:
        arm = m.group(1)
        for col, val in zip(("d", "bytes", "mean_Rstar", "p95_Rstar", "median_Rstar", "candrec100"), m.groups()[1:]):
            rows.append(("bitsvsdims.txt", f"{budget}B.{arm}.{col}", val))

with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["source", "key", "value"]); w.writerows(rows)
print(f"{len(rows)} values -> {out}")
