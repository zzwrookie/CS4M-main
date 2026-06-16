"""Backfill existing OpTC v1.3c full reports with canonical-aware evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline.config.runtime_config import SlimConfig
from scripts.pipeline.outputs.conditional_reports import (
    _phase3g_backfill_result_reports_from_config,
)


SEMANTIC_MODE = "optc_windows_v1_3c_detail"
SEMANTIC_LABEL = "OPTC_WINDOWS_V1_3C_DETAIL"
DEFAULT_DATASETS = ("OPTC_051", "OPTC_201", "OPTC_501")
RESULT_ROOT = "outputs/results/tflr_light/optc_windows_v1_3c_full"
PHASE3E_ROOT = "outputs/cache/phase3e_optc_windows_v1_3c"


def build_optc_v13c_config(dataset: str) -> SlimConfig:
    """Build the minimal config needed to backfill an OpTC v1.3c full result."""
    dataset_name = str(dataset).strip().upper()
    artifact_prefix = f"{dataset_name}_{SEMANTIC_LABEL}_full"
    return SlimConfig(
        dataset=dataset_name,
        out_tag=f"{dataset_name}_{SEMANTIC_LABEL}_FULL_INFER",
        result_root=RESULT_ROOT,
        semantic_mode=SEMANTIC_MODE,
        node_embedding_cache_dir=str(Path(PHASE3E_ROOT) / artifact_prefix / "node"),
        action_embedding_cache_dir=str(Path(PHASE3E_ROOT) / artifact_prefix / "action"),
        event_index_cache_dir=str(Path(PHASE3E_ROOT) / artifact_prefix / "event_index"),
        optc_netflow_node_canonicalization="remote_endpoint_v1_3",
        node_pool_score_mode="base_conf",
    )


def result_dir_for_config(config: SlimConfig) -> Path:
    """Return the existing infer result directory for a backfill config."""
    return Path(config.result_root) / str(config.out_tag)


def backfill_dataset(dataset: str) -> dict[str, object]:
    """Backfill one dataset and return a compact status payload."""
    config = build_optc_v13c_config(dataset)
    result_dir = result_dir_for_config(config)
    if not result_dir.exists():
        raise FileNotFoundError(f"missing infer result directory: {result_dir}")
    eval_path = _phase3g_backfill_result_reports_from_config(
        result_dir=result_dir,
        config=config,
    )
    summary_path = result_dir / "online_event_node_coverage_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "dataset": config.dataset,
        "result_dir": str(result_dir),
        "eval_json": str(eval_path),
        "coverage_summary": str(summary_path),
        "canonical_gt_eval": summary.get("canonical_gt_eval", {}),
    }


def run_backfill(datasets: Sequence[str]) -> list[dict[str, object]]:
    """Backfill each requested dataset in order."""
    return [backfill_dataset(dataset) for dataset in datasets]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    args = parser.parse_args()
    rows = run_backfill(args.datasets)
    print(json.dumps(rows, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
