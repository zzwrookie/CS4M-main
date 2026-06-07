from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from legacy.baselines.semantic_sketch import bucket_rarity, update_bucket_count
from cs4m.utils.common import bucket, stable_hash


@dataclass
class ChainProfileConfig:
    num_buckets: int = 65536
    max_dt_bucket: int = 16
    require_transition: bool = False
    memory_slots: int = 1
    association_weight: float = 0.0


class ChainStreamState:
    def __init__(self) -> None:
        self.last_by_node: dict[int, list[tuple[int, int, int, int]]] = {}


class BenignChainProfile:
    """Benign-only hashed short-chain profile.

    Purpose:
    - Learn compact counts of benign edge/role/transition templates from the
      train stream.
    - Score validation/test events by how rare their current short-chain
      context is under those benign counts.

    Inputs are chronological event rows. The profile never uses labels, ground
    truth node IDs, attack windows, or future test events.
    """

    def __init__(self, config: ChainProfileConfig):
        self.config = config
        buckets = int(config.num_buckets)
        self.edge_counts = np.zeros((buckets,), dtype=np.int32)
        self.role_counts = np.zeros((buckets,), dtype=np.int32)
        self.pair_counts = np.zeros((buckets,), dtype=np.int32)
        self.transition_counts = np.zeros((buckets,), dtype=np.int32)
        self.num_edges = 0
        self.num_transitions = 0
        self._train_state = ChainStreamState()

    @property
    def model_size_bytes(self) -> int:
        return int(
            self.edge_counts.nbytes
            + self.role_counts.nbytes
            + self.pair_counts.nbytes
            + self.transition_counts.nbytes
        )

    @property
    def parameter_count(self) -> int:
        return int(
            self.edge_counts.size
            + self.role_counts.size
            + self.pair_counts.size
            + self.transition_counts.size
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "parameter_count": int(self.parameter_count),
            "model_size_bytes": int(self.model_size_bytes),
            "model_size_mb": float(self.model_size_bytes / (1024.0 * 1024.0)),
            "num_edges": int(self.num_edges),
            "num_transitions": int(self.num_transitions),
            "counts": ["edge", "role", "pair", "transition"],
        }

    def new_state(self) -> ChainStreamState:
        return ChainStreamState()

    def _endpoint_role(self, row: dict[str, Any], node_id: int) -> str:
        if int(node_id) == int(row["src_idx"]):
            return f"{row.get('src_kind', 'unknown')}:{row.get('src_summary', '')}"
        if int(node_id) == int(row["dst_idx"]):
            return f"{row.get('dst_kind', 'unknown')}:{row.get('dst_summary', '')}"
        node_type = int(row.get("info_src_type", 3)) if int(node_id) == int(row.get("info_src", -1)) else int(row.get("info_dst_type", 3))
        return f"type:{node_type}"

    def _templates(self, row: dict[str, Any]) -> tuple[str, str, str, int, int]:
        rel = int(row["relation_id"])
        src_type = int(row["info_src_type"])
        dst_type = int(row["info_dst_type"])
        src_role = self._endpoint_role(row, int(row["info_src"]))
        dst_role = self._endpoint_role(row, int(row["info_dst"]))
        src_kind = str(row.get("src_kind", "unknown"))
        dst_kind = str(row.get("dst_kind", "unknown"))
        action = str(row.get("action", ""))
        obj = str(row.get("object_type", ""))
        role_key = f"R|rel={rel}|types={src_type}>{dst_type}|action={action}|obj={obj}"
        pair_key = f"P|rel={rel}|src={src_role}|dst={dst_role}"
        edge_key = f"E|{role_key}|{pair_key}"
        sig = int(stable_hash(edge_key, seed=911))
        coarse_key = f"C|rel={rel}|types={src_type}>{dst_type}|kinds={src_kind}>{dst_kind}|action={action}|obj={obj}"
        coarse_sig = int(stable_hash(coarse_key, seed=913))
        return edge_key, role_key, pair_key, sig, coarse_sig

    def _dt_bucket(self, now_ns: int, prev_ns: int) -> int:
        if int(prev_ns) < 0:
            return 0
        sec = max(float(int(now_ns) - int(prev_ns)) / 1e9, 0.0)
        return int(min(int(math.log2(sec + 1.0)), int(self.config.max_dt_bucket)))

    def _transition_keys(self, row: dict[str, Any], state: ChainStreamState, coarse_sig: int) -> list[str]:
        ts = int(row["timestamp_ns"])
        rel = int(row["relation_id"])
        out: list[str] = []
        for side, node in (("S", int(row["info_src"])), ("D", int(row["info_dst"]))):
            previous_items = state.last_by_node.get(node)
            if not previous_items:
                continue
            for slot, previous in enumerate(previous_items[: max(int(self.config.memory_slots), 1)]):
                _prev_sig, prev_coarse_sig, prev_rel, prev_ts = previous
                dt_bucket = self._dt_bucket(ts, prev_ts)
                out.append(f"T|side={side}|prev={prev_coarse_sig}|cur={coarse_sig}|rels={prev_rel}>{rel}|dt={dt_bucket}|slot={min(slot, 3)}")
        return out

    def _update_state(self, row: dict[str, Any], state: ChainStreamState, sig: int, coarse_sig: int) -> None:
        ts = int(row["timestamp_ns"])
        rel = int(row["relation_id"])
        state_value = (int(sig), int(coarse_sig), int(rel), int(ts))
        slots = max(int(self.config.memory_slots), 1)
        for node in (int(row["info_src"]), int(row["info_dst"])):
            current = state.last_by_node.get(node, [])
            current.insert(0, state_value)
            del current[slots:]
            state.last_by_node[node] = current

    def profile_row(self, row: dict[str, Any]) -> None:
        edge_key, role_key, pair_key, sig, coarse_sig = self._templates(row)
        update_bucket_count(self.edge_counts, edge_key)
        update_bucket_count(self.role_counts, role_key)
        update_bucket_count(self.pair_counts, pair_key)
        self.num_edges += 1
        for key in self._transition_keys(row, self._train_state, coarse_sig):
            update_bucket_count(self.transition_counts, key)
            self.num_transitions += 1
        self._update_state(row, self._train_state, sig, coarse_sig)

    def score_pre_update(self, row: dict[str, Any], state: ChainStreamState) -> dict[str, float]:
        edge_key, role_key, pair_key, _sig, coarse_sig = self._templates(row)
        edge_rarity = bucket_rarity(self.edge_counts, self.num_edges, edge_key)
        role_rarity = bucket_rarity(self.role_counts, self.num_edges, role_key)
        pair_rarity = bucket_rarity(self.pair_counts, self.num_edges, pair_key)
        transition_values = [
            bucket_rarity(self.transition_counts, self.num_transitions, key)
            for key in self._transition_keys(row, state, coarse_sig)
        ]
        transition_rarity = float(max(transition_values)) if transition_values else 0.0
        transition_mean = float(np.mean(np.asarray(transition_values, dtype=np.float32))) if transition_values else 0.0
        has_transition = 1.0 if transition_values else 0.0
        if transition_values:
            base_score = 0.20 * pair_rarity + 0.10 * edge_rarity + 0.10 * role_rarity + 0.60 * transition_rarity
            association_discrepancy = max(transition_rarity - 0.5 * max(role_rarity, edge_rarity), 0.0)
            association_score = 0.70 * association_discrepancy + 0.20 * transition_mean + 0.10 * pair_rarity
            assoc_weight = min(max(float(self.config.association_weight), 0.0), 1.0)
            score = (1.0 - assoc_weight) * base_score + assoc_weight * association_score
        elif bool(self.config.require_transition):
            association_discrepancy = 0.0
            association_score = 0.0
            score = 0.0
        else:
            association_discrepancy = 0.0
            association_score = 0.0
            score = 0.45 * pair_rarity + 0.15 * edge_rarity + 0.40 * role_rarity
        return {
            "score": float(score),
            "edge_rarity": float(edge_rarity),
            "role_rarity": float(role_rarity),
            "pair_rarity": float(pair_rarity),
            "transition_rarity": float(transition_rarity),
            "transition_mean_rarity": float(transition_mean),
            "association_discrepancy": float(association_discrepancy),
            "association_score": float(association_score),
            "has_transition": float(has_transition),
        }

    def update_state(self, row: dict[str, Any], state: ChainStreamState) -> None:
        _edge_key, _role_key, _pair_key, sig, coarse_sig = self._templates(row)
        self._update_state(row, state, sig, coarse_sig)
