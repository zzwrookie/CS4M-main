#!/usr/bin/env python3
"""TFLR-CSSM unified experimental entry point.

Purpose:
  Provide a paper-facing command for the unified streaming node-alert method.

Inputs/outputs:
  Same DB inputs and result directory outputs as run_tflr_light_db_lowrank.py.
  Enables direct low-rank prediction and online node alerts by default.

How to enable/test:
  python legacy/tools/run_tflr_cssm_unified.py --dataset CLEARSCOPE_E3 --out_tag CSSM_SMOKE --max_test_events 1000

Runtime/memory:
  Same streaming pipeline as the low-rank runner, plus bounded online node risk
  state. The paper-facing profile exposes only a small set of method
  parameters; resource, maintenance, profiling, and cache knobs are fixed
  implementation defaults.

Leakage risk:
  Thresholds use train/validation calibration only. Test labels and final
  rankings are used only after streaming inference for evaluation.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy.tools.run_tflr_light_db_lowrank import main, parse_args


PAPER_PARAMETERS = {
    "rank": 8,
    "online_node_alert_threshold_quantile": 0.999,
    "online_node_risk_decay_sec": 3600.0,
    "online_node_activity_weight": 0.20,
    "online_node_activity_margin": 0.50,
    "online_node_activity_cap": 2.0,
    "active_node_budget": 200000,
    "node_state_ttl_sec": 86400.0,
}

IMPLEMENTATION_DEFAULTS = {
    "latent_dim": 32,
    "max_tokens": 12,
    "vector_cache_size": 200000,
    "semantic_buckets": 131072,
    "identity_buckets": 131072,
    "max_train_samples": 300000,
    "direct_lowrank_iters": 5,
    "direct_lowrank_l2": 0.01,
    "edge_sketch_size": 131072,
    "chain_sketch_size": 65536,
    "memory_prototype_count": 1024,
    "node_state_maintenance_interval": 65536,
    "node_state_eviction_batch": 4096,
    "node_rerank_feature_weights": "0.60,0.10,0.10,0.80,0.90,0.15,0.30,0.90,0.50,1.00,1.20",
    "causal_episode_selective_enabled": False,
    "causal_episode_min_event_tail": 1.0,
    "causal_episode_min_chain": 1.0,
    "causal_episode_min_assoc": 1.0,
    "causal_episode_min_families": 2,
    "profiling_interval_events": 0,
}

FIXED_MAIN_COMPONENTS = {
    "direct_lowrank_enabled": True,
    "online_node_alerts_enabled": True,
    "chain_profile_enabled": True,
    "compact_malicious_chain_enabled": True,
    "causal_episode_support_enabled": False,
    "adaptive_memory_enabled": True,
    "node_rerank_enabled": True,
    "node_rerank_joint_activity_pressure": False,
    "node_rank_mode": "intrinsic_first",
    "evidence_band_gate_enabled": False,
    "mid_rank_gate_enabled": True,
    "local_closure_enabled": False,
    "snc_enabled": False,
    "node_hard_gate_enabled": False,
}


def _provided_flags(argv: list[str]) -> set[str]:
    return {arg.split("=", 1)[0] for arg in argv if arg.startswith("--")}


def _set_if_not_provided(args: object, provided: set[str], name: str, value: object) -> None:
    if f"--{name}" not in provided:
        setattr(args, name, value)


def apply_cssm_canonical_profile(args: object, argv: list[str]) -> object:
    """Apply the paper-facing TFLR-CSSM profile.

    Detection behavior is controlled by a small paper-level set. Capacity,
    sketch, and maintenance values are treated as fixed implementation defaults
    unless explicitly overridden for deployment diagnostics.
    """

    provided = _provided_flags(argv)
    for name, value in PAPER_PARAMETERS.items():
        _set_if_not_provided(args, provided, name, value)
    for name, value in IMPLEMENTATION_DEFAULTS.items():
        _set_if_not_provided(args, provided, name, value)
    _set_if_not_provided(args, provided, "online_node_alert_threshold", -1.0)
    _set_if_not_provided(args, provided, "risk_decay", -1.0)

    for name, value in FIXED_MAIN_COMPONENTS.items():
        setattr(args, name, value)

    args.cssm_unified_entry = True
    args.cssm_config_profile = "evidence_conjunction_tail_rank8"
    args.cssm_paper_parameters = dict(PAPER_PARAMETERS)
    args.cssm_implementation_defaults = dict(IMPLEMENTATION_DEFAULTS)
    args.cssm_fixed_main_components = dict(FIXED_MAIN_COMPONENTS)
    args.cssm_parameter_policy = (
        "Only rank, alert quantile, risk decay, activity regularization, and "
        "state budget are paper-level parameters. The default rank is 8 and "
        "the stored model target is below 5MB. Sketch sizes, disabled "
        "diagnostic causal episode support, direct ALS "
        "iterations, maintenance intervals, profiling, and cache behavior are "
        "fixed implementation defaults, not dataset-tuned detection "
        "hyperparameters. The unified mainline keeps V2 validation-calibrated "
        "online alerts and restores the validated rank-8 tail-suppressor "
        "semantic-stability rerank weight; "
        "episode/subgraph support and joint activity pressure remain available "
        "only as explicit diagnostics because full CLEARSCOPE validation showed "
        "extra runtime without ranking improvement. The rank-16/sketch16 "
        "variant is also diagnostic only: its CLEARSCOPE full run stayed under "
        "10MB but slowed inference and failed to recover the old tail-suppressor "
        "attack-cluster ordering. ActivityNormalizedChainDensity is available "
        "as an explicit candidate-band diagnostic and is disabled by default; "
        "the completed CLEARSCOPE full run showed applied_nodes=0 and no ranking "
        "or online-alert improvement, so it should not be expanded to CADETS/THEIA "
        "without a new predeclared audit."
    )
    return args


if __name__ == "__main__":
    args = parse_args()
    args = apply_cssm_canonical_profile(args, sys.argv[1:])
    main(args)
