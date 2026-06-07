#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs4m.config.provnet_utils import get_indexid2msg, init_database_connection, log
from scripts.data.get_dataset import (
    build_hash_to_type,
    build_hash_uuid_index_map,
    build_uuid_index_map,
    fetch_node_tables,
    get_dataset_splits,
    load_ground_truth_indices,
    parse_split_days,
    use_event_type_filter,
)
from scripts.tools.eval_utils import best_sweep_row, best_under_fp_target, node_confusion_from_masks, target_for_dataset
from scripts.tools.db_stream_utils import _build_index_summaries, _cfg_for_dataset, _query_count, _stream_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute node metrics from an existing ranked CSV with updated relaxed semantics.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--ranked_csv", default="")
    parser.add_argument("--out_json", default="")
    parser.add_argument("--fetch_size", type=int, default=200000)
    parser.add_argument("--max_test_events", type=int, default=0)
    parser.add_argument("--topk_values", default="")
    return parser.parse_args()


def _load_ranked(path: str) -> np.ndarray:
    nodes: list[int] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "node_id" not in (reader.fieldnames or []):
            raise KeyError(f"{path} missing node_id column")
        for row in reader:
            nodes.append(int(row["node_id"]))
    return np.asarray(nodes, dtype=np.int64)


def _default_topk(ranked_len: int) -> list[int]:
    values = [100, 200, 300, 400, 433, 500, 700, 900, 1000, 1200, 1500, 1800, 2000, 2500, 3000, 3500, 4000, 5000, 7000, 10000, 12000, 15000, 20000, 30000, 50000, 100000]
    return [v for v in values if v <= ranked_len] or [min(ranked_len, 100)]


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    result_dir = os.path.abspath(args.result_dir)
    ranked_csv = args.ranked_csv or os.path.join(result_dir, "eval_tflr_lowrank_ranked.csv")
    out_json = args.out_json or os.path.join(result_dir, "eval_tflr_lowrank_relaxed_recount.json")
    ranked = _load_ranked(ranked_csv)
    if ranked.size <= 0:
        raise ValueError(f"No ranked nodes loaded from {ranked_csv}")

    cfg = _cfg_for_dataset(args.dataset)
    conn_cur, conn = init_database_connection(cfg)
    try:
        log("[recount] loading node tables")
        netflow_nodes, process_nodes, file_nodes = fetch_node_tables(conn_cur)
        indexid2summary = _build_index_summaries(get_indexid2msg(conn_cur))
        hash2type = build_hash_to_type(netflow_nodes, process_nodes, file_nodes)
        hash2uuid_index = build_hash_uuid_index_map(netflow_nodes, process_nodes, file_nodes)
        uuid2index = build_uuid_index_map(netflow_nodes, process_nodes, file_nodes)
        max_from_table = max(int(v[1]) for v in hash2uuid_index.values() if v[1] is not None) + 1
        max_node_id = max(max_from_table, int(np.max(ranked)) + 1)
        all_nodes = np.zeros((max_node_id,), dtype=bool)
        positive_nodes = np.zeros((max_node_id,), dtype=bool)
        suspect_nodes = np.zeros((max_node_id,), dtype=bool)

        year_month = cfg.dataset.year_month
        test_days = parse_split_days(get_dataset_splits(cfg, "test"))
        event_filter = use_event_type_filter(cfg)
        test_query_events = _query_count(conn_cur, year_month, test_days, event_filter)
        abnormal_nodes = set(int(x) for x in load_ground_truth_indices(cfg, uuid2index, result_dir))
        processed = 0
        for row in _stream_events(
            conn,
            year_month,
            test_days,
            indexid2summary,
            hash2type,
            hash2uuid_index,
            abnormal_nodes,
            event_filter,
            int(args.fetch_size),
            int(args.max_test_events),
        ):
            label = int(row["label"])
            for node in {int(row["src_idx"]), int(row["dst_idx"]), int(row["info_src"]), int(row["info_dst"])}:
                if 0 <= node < max_node_id:
                    all_nodes[node] = True
                    if int(node) in abnormal_nodes:
                        positive_nodes[node] = True
                    elif label == 1:
                        suspect_nodes[node] = True
            processed += 1
            if processed % 1_000_000 == 0:
                log(f"[recount] processed={processed}")

        topk_values = (
            [int(x.strip()) for x in str(args.topk_values).split(",") if x.strip()]
            if str(args.topk_values).strip()
            else _default_topk(int(ranked.size))
        )
        target = target_for_dataset(args.dataset)
        sweep = []
        for topk in topk_values:
            keep = ranked[: min(int(topk), int(ranked.size))]
            alerted = np.zeros((max_node_id,), dtype=bool)
            alerted[keep[(keep >= 0) & (keep < max_node_id)]] = True
            sweep.append(
                {
                    "topk": int(topk),
                    "num_alerted_nodes": int(keep.shape[0]),
                    "strict_node": node_confusion_from_masks(all_nodes, positive_nodes, suspect_nodes, alerted, relaxed=False),
                    "relaxed_node": node_confusion_from_masks(all_nodes, positive_nodes, suspect_nodes, alerted, relaxed=True),
                }
            )
        best = best_sweep_row(sweep, target)
        best_fp_constrained = best_under_fp_target(sweep, target)
        observed_gt_abnormal_nodes = {
            int(node)
            for node in abnormal_nodes
            if 0 <= int(node) < max_node_id and bool(all_nodes[int(node)])
        }
        unobserved_gt_abnormal_nodes = {int(node) for node in abnormal_nodes} - observed_gt_abnormal_nodes
        summary = {
            "dataset": args.dataset,
            "source_ranked_csv": ranked_csv,
            "source_result_dir": result_dir,
            "recount_semantics": "Relaxed node metrics count only observed GT abnormal nodes as TP/FN. In suspect events, the observed abnormal endpoint can count as TP if alerted; benign endpoints are ignored and never counted as FP. GT nodes absent from the scored test stream under the configured time/event filters are not counted as FN.",
            "test_query_events": int(test_query_events),
            "test_processed": int(processed),
            "num_ranked_nodes": int(ranked.size),
            "node_population": {
                "all_nodes": int(np.sum(all_nodes)),
                "gt_abnormal_nodes_loaded": int(len(abnormal_nodes)),
                "observed_gt_abnormal_nodes": int(len(observed_gt_abnormal_nodes)),
                "unobserved_gt_abnormal_nodes_not_counted_as_fn": int(len(unobserved_gt_abnormal_nodes)),
                "strict_positive_nodes": int(np.sum(positive_nodes)),
                "suspect_only_nodes": int(np.sum(suspect_nodes & ~positive_nodes)),
                "relaxed_positive_nodes": int(np.sum(positive_nodes)),
                "strict_negative_nodes": int(np.sum(all_nodes & ~positive_nodes)),
                "relaxed_negative_nodes": int(np.sum(all_nodes & ~(positive_nodes | suspect_nodes))),
                "definition": "Only observed GT abnormal nodes, i.e. GT nodes that appear in the scored test stream after time/event filters, are eligible for TP/FN.",
            },
            "sweep": sweep,
            "best": best,
            "best_under_fp_target": best_fp_constrained,
            "target": {"dataset": args.dataset, **target},
            "meets_dataset_relaxed_target": bool(
                best and int(best["relaxed_node"]["tp"]) > int(target["relaxed_tp_gt"]) and int(best["relaxed_node"]["fp"]) < int(target["relaxed_fp_lt"])
            ),
            "seconds": float(time.perf_counter() - started),
        }
        with open(out_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print(json.dumps({"out_json": out_json, "best_under_fp_target": best_fp_constrained, "meets_dataset_relaxed_target": summary["meets_dataset_relaxed_target"]}, indent=2, sort_keys=True))
    finally:
        conn_cur.close()
        conn.close()


if __name__ == "__main__":
    main()
