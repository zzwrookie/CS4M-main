#!/usr/bin/env python3
"""
Generate abnormal_nodes.pkl from a GT CSV by remapping UUID -> current index_id.

This script follows the same UUID mapping logic as get_dataset.py:
1) load current node tables from DB
2) build uuid2index map
3) map UUIDs in GT CSV to current index_id
4) save sorted unique index list to abnormal_nodes.pkl
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Iterable, Tuple


def find_repo_root(start: Path) -> Path:
    for cand in [start, *start.parents]:
        if (
            ((cand / "scripts" / "data" / "get_dataset.py").is_file() or (cand / "get_dataset.py").is_file())
            and (cand / "config" / "config.py").is_file()
        ):
            return cand
    raise RuntimeError(f"Cannot locate repo root from {start}")


def detect_separator(csv_path: Path) -> str:
    first = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not first:
        return ","
    line = first[0]
    if "\t" in line and line.count("\t") >= line.count(","):
        return "\t"
    return ","


def load_gt_rows(csv_path: Path) -> Tuple[str, list[list[str]]]:
    sep = detect_separator(csv_path)
    rows: list[list[str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=sep)
        for row in reader:
            if not row:
                continue
            uid = str(row[0]).strip()
            if not uid:
                continue
            rows.append(row)
    return sep, rows


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate abnormal_nodes.pkl for SP25 GT.")
    parser.add_argument(
        "--dataset",
        default="CADETS_E3",
        help="Dataset name used by config/get_dataset runtime args (default: CADETS_E3).",
    )
    parser.add_argument(
        "--input_csv",
        default=str(here / "node_Nginx_Backdoor_06_13.csv"),
        help="GT CSV path with UUID in column 1.",
    )
    parser.add_argument(
        "--out_pkl",
        default=str(here / "abnormal_nodes.pkl"),
        help="Output path for abnormal_nodes.pkl.",
    )
    parser.add_argument(
        "--missing_out",
        default=str(here / "missing_uuids.txt"),
        help="Output path for missing UUID list.",
    )
    parser.add_argument(
        "--rewrite_csv",
        type=int,
        default=1,
        help="If 1, rewrite CSV third column using mapped index_id (default: 1).",
    )
    parser.add_argument(
        "--csv_out",
        default=None,
        help="Output CSV path for rewritten third column (default: overwrite input_csv).",
    )
    parser.add_argument(
        "--fail_on_missing",
        type=int,
        default=0,
        help="If 1, exit non-zero when any UUID cannot be mapped.",
    )
    return parser.parse_args()


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    out_pkl = Path(args.out_pkl).resolve()
    missing_out = Path(args.missing_out).resolve()
    csv_out = Path(args.csv_out).resolve() if args.csv_out else input_csv

    if not input_csv.is_file():
        raise FileNotFoundError(f"input_csv not found: {input_csv}")

    repo_root = find_repo_root(Path(__file__).resolve())
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Import after sys.path is ready.
    from cs4m.config.config import get_runtime_required_args, get_yml_cfg  # type: ignore
    from cs4m.config.provnet_utils import init_database_connection  # type: ignore
    from scripts.data.get_dataset import build_uuid_index_map, fetch_node_tables  # type: ignore

    run_args = get_runtime_required_args(args=[args.dataset])
    cfg = get_yml_cfg(run_args)

    cur, conn = init_database_connection(cfg)
    try:
        netflow_nodes, subject_nodes, file_nodes = fetch_node_tables(cur)
        uuid2index = build_uuid_index_map(netflow_nodes, subject_nodes, file_nodes)
    finally:
        conn.close()

    sep, rows = load_gt_rows(input_csv)
    uuids = [str(r[0]).strip() for r in rows]
    mapped: list[int] = []
    missing: list[str] = []
    rewritten_rows: list[list[str]] = []

    for row, uid in zip(rows, uuids):
        idx = uuid2index.get(uid)
        if idx is None:
            missing.append(uid)
            new_idx = ""
        else:
            new_idx = str(int(idx))
            mapped.append(int(idx))

        info = str(row[1]).strip() if len(row) >= 2 else ""
        rewritten_rows.append([uid, info, new_idx])

    abnormal_nodes = sorted(set(mapped))
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as f:
        pickle.dump(abnormal_nodes, f)

    if missing:
        write_lines(missing_out, sorted(set(missing)))

    if int(args.rewrite_csv) == 1:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with csv_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter=sep)
            writer.writerows(rewritten_rows)

    print(f"[done] input_csv={input_csv}")
    print(f"[done] out_pkl={out_pkl}")
    if int(args.rewrite_csv) == 1:
        print(f"[done] rewritten_csv={csv_out}")
    print(f"[info] gt_rows={len(uuids)} unique_uuid={len(set(uuids))}")
    print(f"[info] mapped_rows={len(mapped)} mapped_unique_index={len(abnormal_nodes)}")
    print(f"[info] missing_rows={len(missing)} missing_unique_uuid={len(set(missing))}")
    if missing:
        print(f"[warn] missing UUID list -> {missing_out}")

    if int(args.fail_on_missing) == 1 and missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
