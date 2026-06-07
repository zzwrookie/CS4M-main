from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import time


OFSM_MODES = {"none", "online_fourier", "online_time_domain", "random"}


@dataclass
class OnlineStateMergeConfig:
    mode: str = "none"
    scope: str = "same_type"
    state_dim: int = 64
    threshold: float = 0.98
    trunc_ratio: float = 0.50
    use_rfft: bool = True
    use_real_only: bool = True
    center_state: bool = False
    normalize: str = "l2"
    eps: float = 1e-8
    min_cluster_size: int = 2
    copy_on_write: bool = True
    random_prob: float = 0.02
    random_seed: int = 0
    diagnostics_max_rows: int = 1_000_000
    match_backend: str = "exact"
    candidate_cap: int = 64
    merge_interval: int = 1
    merge_min_count: int = 1
    diagnostics_enabled: bool = True


@dataclass
class ClusterRecord:
    cluster_id: int
    cluster_type: str
    centroid_state_slot: int
    centroid_signature: np.ndarray
    member_count: int
    representative_node_id: int
    representative_known: bool
    last_update_event_idx: int
    last_merge_similarity: float = 0.0


@dataclass
class NodeRecord:
    node_id: int
    node_type: str
    singleton_slot: int | None = None
    cluster_id: int | None = None
    last_merge_similarity: float = 0.0
    singleton_signature: np.ndarray | None = None


@dataclass
class CopyOnWriteResult:
    node_id: int
    node_type: str
    old_cluster_id: int | None
    copy_on_write: bool
    old_state_slot: int | None
    cluster_size_before: int
    cluster_size_after: int


class OnlineStateMerger:
    """Online OFSM signature and label-free cluster bookkeeping helper."""

    def __init__(self, config: OnlineStateMergeConfig | None = None) -> None:
        self.config = config if config is not None else OnlineStateMergeConfig()
        self.mode = str(self.config.mode)
        if self.mode not in OFSM_MODES:
            raise ValueError(
                "state_merge_mode must be none, online_fourier, online_time_domain, or random",
            )
        if str(self.config.scope) != "same_type":
            raise ValueError("state_merge_scope must be same_type")
        if str(self.config.match_backend) not in {"exact", "bucketed"}:
            raise ValueError("state_merge match_backend must be exact or bucketed")
        self.state_dim = int(self.config.state_dim)
        if self.state_dim <= 0:
            raise ValueError("state_merge state_dim must be positive")
        self.rfft_bins = int(self.state_dim // 2 + 1)
        self.low_bins = int(np.ceil(float(self.config.trunc_ratio) * float(self.rfft_bins)))
        self.low_bins = max(1, min(self.low_bins, self.rfft_bins))
        if self.state_dim == 64 and abs(float(self.config.trunc_ratio) - 0.50) < 1e-12:
            self.low_bins = 17
        self._rng = np.random.default_rng(int(self.config.random_seed))
        self.node_records: dict[int, NodeRecord] = {}
        self.clusters: dict[int, ClusterRecord] = {}
        self._next_cluster_id = 1
        self.num_copy_on_write_total = 0
        self.num_merge_attempts_total = 0
        self.num_merge_success_total = 0
        self.num_remerged_total = 0
        self.num_became_singleton_total = 0
        self.num_random_merge_attempts_total = 0
        self.num_random_merge_success_total = 0
        self.candidate_scan_count_total = 0
        self.candidate_scan_time_sec = 0.0
        self.num_candidates_checked_total = 0
        self.max_candidates_per_attempt = 0
        self.skipped_merge_due_to_interval = 0
        self.skipped_merge_due_to_min_count = 0
        self._singleton_buckets: dict[tuple[str, str, int], set[int]] = {}
        self._cluster_buckets: dict[tuple[str, str, int], set[int]] = {}

    def observe_candidate_scan(self, candidate_count: int, elapsed_sec: float) -> None:
        """Record exact candidate scan cost without changing merge decisions."""
        count = max(int(candidate_count), 0)
        elapsed = max(float(elapsed_sec), 0.0)
        self.candidate_scan_count_total += 1
        self.candidate_scan_time_sec += elapsed
        self.num_candidates_checked_total += count
        self.max_candidates_per_attempt = max(int(self.max_candidates_per_attempt), count)

    def random_value(self) -> float:
        return float(self._rng.random())

    def signature(self, state: np.ndarray) -> np.ndarray:
        vec = np.asarray(state, dtype=np.float32)
        if vec.shape != (self.state_dim,):
            raise ValueError(f"state must have shape ({self.state_dim},)")
        if not bool(np.all(np.isfinite(vec))):
            raise ValueError("state must contain only finite values")
        working = vec.astype(np.float32, copy=True)
        if bool(self.config.center_state):
            working = working - np.float32(np.mean(working))
        if self.mode == "online_fourier":
            spec = np.fft.rfft(working)
            sig = np.real(spec[: self.low_bins]).astype(np.float32, copy=False)
            return self._normalize(sig)
        if self.mode in {"online_time_domain", "random", "none"}:
            return self._normalize(working)
        raise ValueError(f"unsupported state merge mode: {self.mode}")

    def similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        left_sig = (
            self.signature(left)
            if np.asarray(left).shape == (self.state_dim,)
            else np.asarray(left, dtype=np.float32)
        )
        right_sig = (
            self.signature(right)
            if np.asarray(right).shape == (self.state_dim,)
            else np.asarray(right, dtype=np.float32)
        )
        if left_sig.shape != right_sig.shape:
            raise ValueError("signatures must have the same shape")
        return float(np.dot(left_sig, right_sig))

    def register_singleton(
        self,
        node_id: int,
        node_type: str,
        slot: int,
        state: np.ndarray,
        event_idx: int,
    ) -> None:
        node_key = int(node_id)
        old_record = self.node_records.get(node_key)
        if old_record is not None and old_record.singleton_slot is not None:
            self._remove_singleton_bucket(node_key, old_record)
        signature = self.signature(state)
        self.node_records[node_key] = NodeRecord(
            node_id=node_key,
            node_type=self.type_key(node_type),
            singleton_slot=int(slot),
            cluster_id=None,
            last_merge_similarity=self.last_similarity_for_node(node_key),
            singleton_signature=signature,
        )
        self._add_singleton_bucket(node_key, self.node_records[node_key])

    def unregister_node(self, node_id: int) -> None:
        node_key = int(node_id)
        record = self.node_records.get(node_key)
        if record is not None and record.singleton_slot is not None:
            self._remove_singleton_bucket(node_key, record)
        self.node_records.pop(node_key, None)

    def cluster(self, cluster_id: int) -> ClusterRecord:
        return self.clusters[int(cluster_id)]

    def cluster_id_for_node(self, node_id: int) -> int | None:
        record = self.node_records.get(int(node_id))
        if record is None:
            return None
        return record.cluster_id

    def cluster_for_node(self, node_id: int) -> ClusterRecord | None:
        cluster_id = self.cluster_id_for_node(node_id)
        if cluster_id is None:
            return None
        return self.clusters.get(int(cluster_id))

    def singleton_slot_for_node(self, node_id: int) -> int | None:
        record = self.node_records.get(int(node_id))
        if record is None:
            return None
        return record.singleton_slot

    def physical_slot_for_node(self, node_id: int) -> int | None:
        record = self.node_records.get(int(node_id))
        if record is None:
            return None
        if record.cluster_id is not None:
            cluster = self.clusters.get(int(record.cluster_id))
            return None if cluster is None else int(cluster.centroid_state_slot)
        return record.singleton_slot

    def is_clustered(self, node_id: int) -> bool:
        return self.cluster_id_for_node(node_id) is not None

    def last_similarity_for_node(self, node_id: int) -> float:
        record = self.node_records.get(int(node_id))
        if record is None:
            return 0.0
        return float(record.last_merge_similarity)

    def create_cluster_from_singletons(
        self,
        node_id: int,
        candidate_node_id: int,
        cluster_slot: int,
        centroid_state: np.ndarray,
        event_idx: int,
        similarity: float = 1.0,
    ) -> int:
        node_key = int(node_id)
        candidate_key = int(candidate_node_id)
        node_record = self.node_records[node_key]
        candidate_record = self.node_records[candidate_key]
        cluster_id = int(self._next_cluster_id)
        self._next_cluster_id += 1
        cluster_type = self.type_key(node_record.node_type)
        self.clusters[cluster_id] = ClusterRecord(
            cluster_id=cluster_id,
            cluster_type=cluster_type,
            centroid_state_slot=int(cluster_slot),
            centroid_signature=self.signature(centroid_state),
            member_count=2,
            representative_node_id=node_key,
            representative_known=True,
            last_update_event_idx=int(event_idx),
            last_merge_similarity=float(similarity),
        )
        self._remove_singleton_bucket(node_key, node_record)
        self._remove_singleton_bucket(candidate_key, candidate_record)
        node_record.cluster_id = cluster_id
        node_record.singleton_slot = None
        node_record.singleton_signature = None
        node_record.last_merge_similarity = float(similarity)
        candidate_record.cluster_id = cluster_id
        candidate_record.singleton_slot = None
        candidate_record.singleton_signature = None
        candidate_record.last_merge_similarity = float(similarity)
        self._add_cluster_bucket(cluster_id, self.clusters[cluster_id])
        self.num_merge_success_total += 1
        self.num_remerged_total += 1
        return cluster_id

    def join_cluster(
        self,
        node_id: int,
        cluster_id: int,
        cluster_slot: int,
        centroid_state: np.ndarray,
        event_idx: int,
        similarity: float,
    ) -> None:
        node_key = int(node_id)
        record = self.node_records[node_key]
        cluster = self.clusters[int(cluster_id)]
        self._remove_cluster_bucket(int(cluster_id), cluster)
        self._remove_singleton_bucket(node_key, record)
        record.cluster_id = int(cluster_id)
        record.singleton_slot = None
        record.singleton_signature = None
        record.last_merge_similarity = float(similarity)
        cluster.centroid_state_slot = int(cluster_slot)
        cluster.centroid_signature = self.signature(centroid_state)
        cluster.member_count += 1
        cluster.last_update_event_idx = int(event_idx)
        cluster.last_merge_similarity = float(similarity)
        if not bool(cluster.representative_known):
            cluster.representative_node_id = node_key
            cluster.representative_known = True
        self._add_cluster_bucket(int(cluster_id), cluster)
        self.num_merge_success_total += 1
        self.num_remerged_total += 1

    def update_cluster_centroid(
        self,
        cluster_id: int,
        cluster_slot: int,
        centroid_state: np.ndarray,
        event_idx: int,
    ) -> None:
        cluster = self.clusters[int(cluster_id)]
        self._remove_cluster_bucket(int(cluster_id), cluster)
        cluster.centroid_state_slot = int(cluster_slot)
        cluster.centroid_signature = self.signature(centroid_state)
        cluster.last_update_event_idx = int(event_idx)
        self._add_cluster_bucket(int(cluster_id), cluster)

    def peel_node_from_cluster(self, node_id: int, event_idx: int) -> CopyOnWriteResult:
        node_key = int(node_id)
        record = self.node_records.get(node_key)
        if record is None or record.cluster_id is None:
            return CopyOnWriteResult(
                node_id=node_key,
                node_type="" if record is None else str(record.node_type),
                old_cluster_id=None,
                copy_on_write=False,
                old_state_slot=None,
                cluster_size_before=1,
                cluster_size_after=1,
            )
        cluster_id = int(record.cluster_id)
        cluster = self.clusters[cluster_id]
        before = int(cluster.member_count)
        cluster.member_count = max(0, int(cluster.member_count) - 1)
        after = int(cluster.member_count)
        if int(cluster.representative_node_id) == node_key and after > 0:
            cluster.representative_node_id = -1
            cluster.representative_known = False
        record.cluster_id = None
        record.singleton_slot = None
        record.singleton_signature = None
        self.num_copy_on_write_total += 1
        return CopyOnWriteResult(
            node_id=node_key,
            node_type=str(record.node_type),
            old_cluster_id=cluster_id,
            copy_on_write=True,
            old_state_slot=int(cluster.centroid_state_slot),
            cluster_size_before=before,
            cluster_size_after=after,
        )

    def remove_empty_cluster(self, cluster_id: int) -> None:
        cluster = self.clusters.get(int(cluster_id))
        if cluster is not None and int(cluster.member_count) <= 0:
            self._remove_cluster_bucket(int(cluster_id), cluster)
            self.clusters.pop(int(cluster_id), None)

    def bucketed_candidate_refs(
        self,
        *,
        node_id: int,
        node_type: str,
        signature: np.ndarray,
    ) -> list[tuple[str, int]]:
        """Return bounded approximate same-type candidate refs for bucketed matching."""
        if str(self.config.match_backend) != "bucketed":
            return []
        type_key = self.type_key(node_type)
        buckets = self._neighbor_bucket_ids(signature)
        refs: list[tuple[str, int, int, int]] = []
        for bucket_id in buckets:
            key = self._bucket_key(type_key, bucket_id)
            for cluster_id in self._cluster_buckets.get(key, set()):
                cluster = self.clusters.get(int(cluster_id))
                if cluster is None or str(cluster.cluster_type) != type_key:
                    continue
                refs.append(
                    (
                        "cluster",
                        int(cluster_id),
                        int(cluster.last_update_event_idx),
                        int(cluster.member_count),
                    ),
                )
            for other_id in self._singleton_buckets.get(key, set()):
                if int(other_id) == int(node_id):
                    continue
                record = self.node_records.get(int(other_id))
                if (
                    record is None
                    or str(record.node_type) != type_key
                    or record.singleton_slot is None
                ):
                    continue
                refs.append(("singleton", int(other_id), 0, 1))
        refs.sort(key=lambda item: (item[2], item[3], item[1]), reverse=True)
        cap = max(int(self.config.candidate_cap), 1)
        return [(kind, ident) for kind, ident, _, _ in refs[:cap]]

    def replace_slot(self, old_slot: int, new_slot: int) -> None:
        old_key = int(old_slot)
        new_key = int(new_slot)
        for record in self.node_records.values():
            if record.singleton_slot == old_key:
                record.singleton_slot = new_key
        for cluster in self.clusters.values():
            if int(cluster.centroid_state_slot) == old_key:
                cluster.centroid_state_slot = new_key

    def profile_stats(self) -> dict[str, int | float | str]:
        logical_node_count = int(len(self.node_records))
        num_clusters = int(len(self.clusters))
        num_clustered_nodes = sum(
            1 for record in self.node_records.values() if record.cluster_id is not None
        )
        num_singleton_states = sum(
            1 for record in self.node_records.values() if record.singleton_slot is not None
        )
        member_counts = [int(cluster.member_count) for cluster in self.clusters.values()]
        num_physical_states = int(num_singleton_states + num_clusters)
        compression_ratio = (
            float(logical_node_count) / float(max(num_physical_states, 1))
            if logical_node_count
            else 0.0
        )
        return {
            "state_merge_mode": str(self.mode),
            "state_merge_scope": str(self.config.scope),
            "merge_threshold": float(self.config.threshold),
            "trunc_ratio": float(self.config.trunc_ratio),
            "random_prob": float(self.config.random_prob),
            "random_seed": int(self.config.random_seed),
            "logical_node_count": logical_node_count,
            "num_physical_states": num_physical_states,
            "num_singleton_states": int(num_singleton_states),
            "num_clusters": num_clusters,
            "num_clustered_nodes": int(num_clustered_nodes),
            "compression_ratio": float(compression_ratio),
            "avg_cluster_size": float(np.mean(member_counts)) if member_counts else 0.0,
            "max_cluster_size": int(max(member_counts, default=0)),
            "num_copy_on_write_total": int(self.num_copy_on_write_total),
            "num_merge_attempts_total": int(self.num_merge_attempts_total),
            "num_merge_success_total": int(self.num_merge_success_total),
            "num_remerged_total": int(self.num_remerged_total),
            "num_became_singleton_total": int(self.num_became_singleton_total),
            "num_random_merge_attempts_total": int(self.num_random_merge_attempts_total),
            "num_random_merge_success_total": int(self.num_random_merge_success_total),
            "candidate_scan_count_total": int(self.candidate_scan_count_total),
            "candidate_scan_time_sec": float(self.candidate_scan_time_sec),
            "num_candidates_checked_total": int(self.num_candidates_checked_total),
            "avg_candidates_per_attempt": (
                float(self.num_candidates_checked_total)
                / float(max(self.candidate_scan_count_total, 1))
            ),
            "max_candidates_per_attempt": int(self.max_candidates_per_attempt),
            "match_backend": str(self.config.match_backend),
            "candidate_cap": int(self.config.candidate_cap),
            "merge_interval": int(self.config.merge_interval),
            "merge_min_count": int(self.config.merge_min_count),
            "skipped_merge_due_to_interval": int(self.skipped_merge_due_to_interval),
            "skipped_merge_due_to_min_count": int(self.skipped_merge_due_to_min_count),
        }

    @staticmethod
    def type_key(node_type: object) -> str:
        text = "" if node_type is None else str(node_type).strip().lower()
        for key in ("process", "netflow", "file"):
            if key in text:
                return key
        return "unknown"

    def _normalize(self, value: np.ndarray) -> np.ndarray:
        if str(self.config.normalize) != "l2":
            raise ValueError("state_merge normalize must be l2")
        norm = float(np.linalg.norm(value))
        return (value / np.float32(norm + float(self.config.eps))).astype(
            np.float32,
            copy=False,
        )

    def _signature_bucket(self, signature: np.ndarray) -> int:
        sig = np.asarray(signature, dtype=np.float32)
        if sig.size == 0:
            return 0
        return int(np.argmax(np.abs(sig)))

    def _neighbor_bucket_ids(self, signature: np.ndarray) -> list[int]:
        center = self._signature_bucket(signature)
        upper = int(self.low_bins if self.mode == "online_fourier" else self.state_dim)
        candidates = [center - 1, center, center + 1]
        return [int(value) for value in candidates if 0 <= int(value) < upper]

    def _bucket_key(self, node_type: str, bucket_id: int) -> tuple[str, str, int]:
        return (self.type_key(node_type), str(self.mode), int(bucket_id))

    def _add_singleton_bucket(self, node_id: int, record: NodeRecord) -> None:
        if record.singleton_signature is None:
            return
        key = self._bucket_key(record.node_type, self._signature_bucket(record.singleton_signature))
        self._singleton_buckets.setdefault(key, set()).add(int(node_id))

    def _remove_singleton_bucket(self, node_id: int, record: NodeRecord) -> None:
        if record.singleton_signature is None:
            return
        key = self._bucket_key(record.node_type, self._signature_bucket(record.singleton_signature))
        bucket = self._singleton_buckets.get(key)
        if bucket is not None:
            bucket.discard(int(node_id))

    def _add_cluster_bucket(self, cluster_id: int, cluster: ClusterRecord) -> None:
        key = self._bucket_key(
            cluster.cluster_type,
            self._signature_bucket(cluster.centroid_signature),
        )
        self._cluster_buckets.setdefault(key, set()).add(int(cluster_id))

    def _remove_cluster_bucket(self, cluster_id: int, cluster: ClusterRecord) -> None:
        key = self._bucket_key(
            cluster.cluster_type,
            self._signature_bucket(cluster.centroid_signature),
        )
        bucket = self._cluster_buckets.get(key)
        if bucket is not None:
            bucket.discard(int(cluster_id))
