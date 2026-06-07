from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import time

import numpy as np

from cs4m.state.online_state_merging import OnlineStateMergeConfig, OnlineStateMerger


TYPE_PRIORITY = {"unknown": 0, "netflow": 1, "file": 2, "process": 3}


@dataclass
class ProbationaryNode:
    node_type: str
    count: int
    first_seen: int
    last_seen: int
    as_src_count: int
    as_dst_count: int
    last_score: float
    recent_score_max: float
    ever_alerted: bool
    pending_state: np.ndarray


class ProbationaryLRUStateMemory:
    """Bounded active node memory with file/netflow probationary admission."""

    def __init__(
        self,
        state_dim: int,
        active_max_nodes: int,
        probationary_max_nodes: int,
        probationary_min_count: int,
        lru_evict_batch: int = 10000,
        alert_protect_events: int = 100000,
        state_merge_config: OnlineStateMergeConfig | None = None,
    ) -> None:
        self.state_dim = int(state_dim)
        self.active_max_nodes = int(active_max_nodes)
        self.probationary_max_nodes = int(probationary_max_nodes)
        self.probationary_min_count = int(probationary_min_count)
        self.lru_evict_batch = int(lru_evict_batch)
        self.alert_protect_events = int(alert_protect_events)
        self.node_to_slot: dict[int, int] = {}
        self._state_array = np.zeros((0, self.state_dim), dtype=np.float32)
        self._node_ids = np.zeros((0,), dtype=np.int64)
        self._type_names: list[str] = []
        self._last_touch = np.zeros((0,), dtype=np.int64)
        self._touch_count = np.zeros((0,), dtype=np.int32)
        self._protect_until = np.zeros((0,), dtype=np.int64)
        self._state_size = 0
        self._clock = 0
        self._probationary: dict[int, ProbationaryNode] = {}
        self.promoted_nodes_total = 0
        self.evicted_active_nodes_total = 0
        self.evicted_probationary_nodes_total = 0
        self.state_merge_config = state_merge_config or OnlineStateMergeConfig(
            mode="none",
            state_dim=self.state_dim,
        )
        self.state_merger = (
            OnlineStateMerger(self.state_merge_config)
            if str(self.state_merge_config.mode) != "none"
            else None
        )
        self.last_merge_actions: list[dict[str, Any]] = []
        self._last_merge_metadata: dict[str, Any] = self._empty_merge_metadata()

    @property
    def active_count(self) -> int:
        return int(self._state_size)

    @property
    def probationary_count(self) -> int:
        return int(len(self._probationary))

    @property
    def state_array(self) -> np.ndarray:
        return self._state_array[: self._state_size]

    @property
    def state_array_mb(self) -> float:
        return float(self._state_array.nbytes / (1024.0 * 1024.0))

    def reset(self) -> None:
        """Clear all online state and counters."""
        self.__init__(
            self.state_dim,
            self.active_max_nodes,
            self.probationary_max_nodes,
            self.probationary_min_count,
            self.lru_evict_batch,
            self.alert_protect_events,
            self.state_merge_config,
        )

    def read_state(self, node_id: int, node_type: str) -> np.ndarray:
        """Read a node's active state, or zero for unseen/probationary nodes."""
        self._clock += 1
        if self.state_merger is not None:
            state = self._read_state_merged(int(node_id), node_type)
            if state is not None:
                return state
        slot = self.node_to_slot.get(int(node_id))
        if slot is None:
            type_key = self._type_key(node_type)
            if type_key == "process":
                return np.zeros((self.state_dim,), dtype=np.float32)
            self._ensure_probationary(int(node_id), type_key, count_event=False)
            self._evict_probationary_if_needed()
            return np.zeros((self.state_dim,), dtype=np.float32)
        self._touch_slot(int(slot))
        return self._state_array[int(slot)].astype(np.float32, copy=True)

    def update_state(
        self,
        node_id: int,
        node_type: str,
        message: np.ndarray,
        state_model: Any,
        q_t: float,
        protected: bool,
        current_node_ids: Iterable[int] | None = None,
        role: str = "",
        score: float | None = None,
    ) -> None:
        """Apply one post-score update with probationary admission and LRU bounds."""
        self._clock += 1
        self.last_merge_actions = []
        self._last_merge_metadata = self._empty_merge_metadata()
        node_key = int(node_id)
        type_key = self._type_key(node_type)
        msg = np.asarray(message, dtype=np.float32)
        if msg.shape != (self.state_dim,):
            raise ValueError(f"message must have shape ({self.state_dim},)")
        if not bool(np.all(np.isfinite(msg))):
            raise ValueError("message must contain only finite values")
        if self.state_merger is not None:
            self._update_state_merged(
                node_key,
                type_key,
                msg,
                state_model,
                q_t=q_t,
                protected=protected,
                current_node_ids=current_node_ids,
                role=role,
                score=score,
            )
            return
        if node_key in self.node_to_slot:
            slot = int(self.node_to_slot[node_key])
            self._state_array[slot] = state_model.step(self._state_array[slot], msg, q_t=q_t)
            self._touch_slot(slot)
            if protected:
                self._protect_until[slot] = max(
                    int(self._protect_until[slot]),
                    int(self._clock + self.alert_protect_events),
                )
            return
        if type_key == "process" or self.probationary_min_count <= 1:
            slot = self._activate(node_key, type_key, current_node_ids=current_node_ids)
            self._state_array[slot] = state_model.step(self._state_array[slot], msg, q_t=q_t)
            if self.state_merger is not None:
                self.state_merger.register_singleton(
                    node_key,
                    type_key,
                    slot,
                    self._state_array[slot],
                    int(self._clock),
                )
            if protected:
                self._protect_until[slot] = int(self._clock + self.alert_protect_events)
            return
        probationary = self._ensure_probationary(node_key, type_key, count_event=False)
        probationary.count += 1
        probationary.last_seen = int(self._clock)
        role_key = str(role).strip().lower()
        if role_key == "src":
            probationary.as_src_count += 1
        elif role_key == "dst":
            probationary.as_dst_count += 1
        if score is not None and np.isfinite(float(score)):
            score_value = float(score)
            probationary.last_score = score_value
            probationary.recent_score_max = max(float(probationary.recent_score_max), score_value)
        if bool(protected):
            probationary.ever_alerted = True
        probationary.pending_state = state_model.step(
            probationary.pending_state,
            msg,
            q_t=q_t,
        )
        if self._should_promote(probationary):
            slot = self._activate(node_key, type_key, current_node_ids=current_node_ids)
            self._state_array[slot] = probationary.pending_state
            if self.state_merger is not None:
                self.state_merger.register_singleton(
                    node_key,
                    type_key,
                    slot,
                    self._state_array[slot],
                    int(self._clock),
                )
            if protected:
                self._protect_until[slot] = int(self._clock + self.alert_protect_events)
            self._probationary.pop(node_key, None)
            self.promoted_nodes_total += 1
        self._evict_probationary_if_needed()

    def mark_protected(self, node_id: int, state_model: Any | None = None) -> None:
        """Protect an active node from normal eviction for the configured event horizon."""
        node_key = int(node_id)
        slot = self.node_to_slot.get(node_key)
        if slot is not None:
            self._protect_until[int(slot)] = int(self._clock + self.alert_protect_events)
            return
        probationary = self._probationary.get(node_key)
        if probationary is None:
            return
        probationary.ever_alerted = True
        if state_model is not None and self._should_promote(probationary):
            slot = self._activate(
                node_key,
                str(probationary.node_type),
                current_node_ids={node_key},
            )
            self._state_array[slot] = probationary.pending_state
            self._protect_until[int(slot)] = int(self._clock + self.alert_protect_events)
            self._probationary.pop(node_key, None)
            self.promoted_nodes_total += 1

    def stats(self) -> dict[str, int | float | str]:
        """Return label-free memory diagnostics."""
        memory_bytes = (
            self._state_array.nbytes
            + self._node_ids.nbytes
            + self._last_touch.nbytes
            + self._touch_count.nbytes
            + self._protect_until.nbytes
        )
        probationary_event_count_total = sum(
            int(node.count) for node in self._probationary.values()
        )
        probationary_as_src_count_total = sum(
            int(node.as_src_count) for node in self._probationary.values()
        )
        probationary_as_dst_count_total = sum(
            int(node.as_dst_count) for node in self._probationary.values()
        )
        return {
            "state_memory_policy": "probationary_lru",
            "active_nodes": int(self.active_count),
            "probationary_nodes": int(self.probationary_count),
            "probationary_event_count_total": int(probationary_event_count_total),
            "probationary_as_src_count_total": int(probationary_as_src_count_total),
            "probationary_as_dst_count_total": int(probationary_as_dst_count_total),
            "promoted_nodes_total": int(self.promoted_nodes_total),
            "evicted_active_nodes_total": int(self.evicted_active_nodes_total),
            "evicted_probationary_nodes_total": int(self.evicted_probationary_nodes_total),
            "state_memory_mb_est": float(memory_bytes / (1024.0 * 1024.0)),
            "state_slots": int(self.active_count),
            "state_array_mb": float(self.state_array_mb),
            **self.state_merge_profile_stats(),
        }

    def storage_audit(
        self,
        *,
        state_memory_mode: str = "bounded",
        state_memory_policy: str = "probationary_lru",
    ) -> dict[str, int | float | str | bool | list[int]]:
        """Return storage-level diagnostics for the active/probationary memory."""
        allocated_slots = int(self._state_array.shape[0])
        active_count = int(self.active_count)
        slot_utilization = (
            float(active_count) / float(max(allocated_slots, 1))
            if allocated_slots > 0
            else 0.0
        )
        return {
            "state_memory_mode": str(state_memory_mode),
            "state_memory_policy": str(state_memory_policy),
            "uses_active_state_memory": True,
            "uses_probationary_memory": True,
            "uses_bounded_lru": True,
            "physical_storage_type": "numpy_growing_slot_array",
            "state_storage_mode": "lazy_active_existing",
            "is_dense_preallocated": False,
            "is_lazy_allocated": True,
            "state_array_shape": [int(value) for value in self._state_array.shape],
            "state_dim": int(self.state_dim),
            "state_dtype": str(self._state_array.dtype),
            "allocated_state_slots": allocated_slots,
            "active_state_count": active_count,
            "probationary_state_count": int(self.probationary_count),
            "max_reserved_slots": int(self.active_max_nodes),
            "state_table_mb": float(self.state_array_mb),
            "slot_utilization": float(slot_utilization),
            "allocation_trigger": "node_activation_or_probationary_promotion",
            "eviction_policy": "bounded_lru_low_priority_unprotected",
            "ttl_policy": "none",
        }

    def _type_key(self, node_type: object) -> str:
        text = "" if node_type is None else str(node_type).strip().lower()
        for key in ("process", "netflow", "file"):
            if key in text:
                return key
        return "unknown"

    def _ensure_probationary(
        self,
        node_id: int,
        node_type: str,
        count_event: bool,
    ) -> ProbationaryNode:
        node_key = int(node_id)
        probationary = self._probationary.get(node_key)
        if probationary is None:
            probationary = ProbationaryNode(
                node_type=str(node_type),
                count=0,
                first_seen=int(self._clock),
                last_seen=int(self._clock),
                as_src_count=0,
                as_dst_count=0,
                last_score=0.0,
                recent_score_max=0.0,
                ever_alerted=False,
                pending_state=np.zeros((self.state_dim,), dtype=np.float32),
            )
            self._probationary[node_key] = probationary
        else:
            probationary.node_type = str(node_type)
            probationary.last_seen = int(self._clock)
        if bool(count_event):
            probationary.count += 1
        return probationary

    def _should_promote(self, probationary: ProbationaryNode) -> bool:
        if str(probationary.node_type) == "process":
            return True
        if int(probationary.count) >= int(self.probationary_min_count):
            return True
        if int(probationary.as_src_count) >= 1:
            return True
        if bool(probationary.ever_alerted):
            return True
        return False

    def _ensure_capacity(self, required: int) -> None:
        if required <= int(self._state_array.shape[0]):
            return
        current = int(self._state_array.shape[0])
        new_capacity = max(required, max(16, current * 2))
        new_array = np.zeros((new_capacity, self.state_dim), dtype=np.float32)
        if self._state_size:
            new_array[: self._state_size] = self._state_array[: self._state_size]
        self._state_array = new_array
        self._node_ids = self._grow_1d_array(self._node_ids, new_capacity, np.int64)
        self._last_touch = self._grow_1d_array(self._last_touch, new_capacity, np.int64)
        self._touch_count = self._grow_1d_array(self._touch_count, new_capacity, np.int32)
        self._protect_until = self._grow_1d_array(self._protect_until, new_capacity, np.int64)
        while len(self._type_names) < new_capacity:
            self._type_names.append("unknown")

    def _activate(
        self,
        node_id: int,
        node_type: str,
        current_node_ids: Iterable[int] | None,
    ) -> int:
        if self.active_max_nodes > 0 and self._state_size >= self.active_max_nodes:
            self._evict_active_if_needed(current_node_ids or ())
        if self.active_max_nodes > 0 and self._state_size >= self.active_max_nodes:
            self.evicted_active_nodes_total += 0
            raise RuntimeError("SSPM active state memory is full and no slot is evictable")
        slot = int(self._state_size)
        self._ensure_capacity(slot + 1)
        self.node_to_slot[int(node_id)] = slot
        self._node_ids[slot] = int(node_id)
        self._type_names[slot] = str(node_type)
        self._last_touch[slot] = int(self._clock)
        self._touch_count[slot] = np.int32(0)
        self._protect_until[slot] = np.int64(0)
        self._state_array[slot].fill(0.0)
        self._state_size += 1
        return slot

    def _touch_slot(self, slot: int) -> None:
        self._last_touch[int(slot)] = int(self._clock)
        current = int(self._touch_count[int(slot)])
        self._touch_count[int(slot)] = np.int32(min(current + 1, np.iinfo(np.int32).max))

    def last_merge_metadata(self) -> dict[str, Any]:
        """Return label-free OFSM metadata for the most recent memory update."""
        return {
            **self._last_merge_metadata,
            "actions": list(self._last_merge_metadata.get("actions", [])),
        }

    def state_merge_profile_stats(self) -> dict[str, int | float | str]:
        """Return label-free OFSM compression counters."""
        if self.state_merger is not None:
            return self.state_merger.profile_stats()
        logical_node_count = int(self.active_count)
        compression_ratio = 1.0 if logical_node_count else 0.0
        return {
            "state_merge_mode": "none",
            "state_merge_scope": str(self.state_merge_config.scope),
            "merge_threshold": float(self.state_merge_config.threshold),
            "trunc_ratio": float(self.state_merge_config.trunc_ratio),
            "random_prob": float(self.state_merge_config.random_prob),
            "random_seed": int(self.state_merge_config.random_seed),
            "logical_node_count": logical_node_count,
            "num_physical_states": logical_node_count,
            "num_singleton_states": logical_node_count,
            "num_clusters": 0,
            "num_clustered_nodes": 0,
            "compression_ratio": float(compression_ratio),
            "avg_cluster_size": 0.0,
            "max_cluster_size": 0,
            "num_copy_on_write_total": 0,
            "num_merge_attempts_total": 0,
            "num_merge_success_total": 0,
            "num_remerged_total": 0,
            "num_became_singleton_total": 0,
            "num_random_merge_attempts_total": 0,
            "num_random_merge_success_total": 0,
            "candidate_scan_count_total": 0,
            "candidate_scan_time_sec": 0.0,
            "num_candidates_checked_total": 0,
            "avg_candidates_per_attempt": 0.0,
            "max_candidates_per_attempt": 0,
            "match_backend": str(self.state_merge_config.match_backend),
            "candidate_cap": int(self.state_merge_config.candidate_cap),
            "merge_interval": int(self.state_merge_config.merge_interval),
            "merge_min_count": int(self.state_merge_config.merge_min_count),
            "skipped_merge_due_to_interval": 0,
            "skipped_merge_due_to_min_count": 0,
        }

    def state_merge_node_snapshot(self, node_id: int, node_type: str = "") -> dict[str, Any]:
        """Return an online-visible OFSM snapshot for one node."""
        mode = str(self.state_merge_config.mode)
        if self.state_merger is None:
            return self._empty_node_snapshot(mode=mode)
        node_key = int(node_id)
        cluster = self.state_merger.cluster_for_node(node_key)
        if cluster is not None:
            return {
                "cluster_id": int(cluster.cluster_id),
                "cluster_size": int(cluster.member_count),
                "merge_similarity": float(self.state_merger.last_similarity_for_node(node_key)),
                "is_clustered": 1,
                "merge_mode": mode,
                "physical_state_id": int(cluster.centroid_state_slot),
            }
        slot = self.state_merger.singleton_slot_for_node(node_key)
        if slot is not None:
            return {
                "cluster_id": "",
                "cluster_size": 1,
                "merge_similarity": float(self.state_merger.last_similarity_for_node(node_key)),
                "is_clustered": 0,
                "merge_mode": mode,
                "physical_state_id": int(slot),
            }
        return self._empty_node_snapshot(mode=mode)

    def _empty_node_snapshot(self, mode: str) -> dict[str, Any]:
        return {
            "cluster_id": "",
            "cluster_size": 1,
            "merge_similarity": 0.0,
            "is_clustered": 0,
            "merge_mode": str(mode),
            "physical_state_id": "",
        }

    def _empty_merge_metadata(self) -> dict[str, Any]:
        return {
            "mode": str(self.state_merge_config.mode),
            "actions": [],
            "node": {},
            "src": {},
            "dst": {},
            "diagnostics_truncated": False,
        }

    def _read_state_merged(self, node_id: int, node_type: str) -> np.ndarray | None:
        if self.state_merger is None:
            return None
        node_key = int(node_id)
        cluster = self.state_merger.cluster_for_node(node_key)
        if cluster is not None:
            slot = int(cluster.centroid_state_slot)
            self._touch_slot(slot)
            return self._state_array[slot].astype(np.float32, copy=True)
        slot = self.state_merger.singleton_slot_for_node(node_key)
        if slot is None:
            return None
        self._touch_slot(int(slot))
        return self._state_array[int(slot)].astype(np.float32, copy=True)

    def _update_state_merged(
        self,
        node_id: int,
        node_type: str,
        message: np.ndarray,
        state_model: Any,
        q_t: float,
        protected: bool,
        current_node_ids: Iterable[int] | None,
        role: str,
        score: float | None,
    ) -> None:
        if self.state_merger is None:
            raise RuntimeError("OFSM update called while state merger is disabled")
        old_state, slot, cow_action = self._prepare_merged_update_slot(
            node_id=node_id,
            node_type=node_type,
            message=message,
            state_model=state_model,
            q_t=q_t,
            protected=protected,
            current_node_ids=current_node_ids,
            role=role,
            score=score,
        )
        if slot is None:
            return
        if cow_action is not None:
            self._append_merge_action(cow_action)
        if old_state is not None:
            self._state_array[int(slot)] = state_model.step(old_state, message, q_t=q_t)
        self._touch_slot(int(slot))
        if protected:
            self._protect_until[int(slot)] = max(
                int(self._protect_until[int(slot)]),
                int(self._clock + self.alert_protect_events),
            )
        self.state_merger.register_singleton(
            int(node_id),
            node_type,
            int(slot),
            self._state_array[int(slot)],
            int(self._clock),
        )
        self._try_merge_updated_singleton(
            node_id=int(node_id),
            node_type=node_type,
            slot=int(slot),
            protected=protected,
            role=role,
            score=score,
            cow_action=cow_action,
        )
        snapshot = self.state_merge_node_snapshot(int(node_id), node_type)
        self._last_merge_metadata = self._empty_merge_metadata()
        self._last_merge_metadata["node"] = snapshot
        role_key = str(role).strip().lower()
        if role_key in {"src", "dst"}:
            self._last_merge_metadata[role_key] = snapshot
        self._last_merge_metadata["actions"] = list(self.last_merge_actions)

    def _prepare_merged_update_slot(
        self,
        node_id: int,
        node_type: str,
        message: np.ndarray,
        state_model: Any,
        q_t: float,
        protected: bool,
        current_node_ids: Iterable[int] | None,
        role: str,
        score: float | None,
    ) -> tuple[np.ndarray | None, int | None, dict[str, Any] | None]:
        if self.state_merger is None:
            raise RuntimeError("OFSM slot preparation called while state merger is disabled")
        node_key = int(node_id)
        cluster = self.state_merger.cluster_for_node(node_key)
        if cluster is not None:
            old_slot = int(cluster.centroid_state_slot)
            old_state = self._state_array[old_slot].astype(np.float32, copy=True)
            cow = self.state_merger.peel_node_from_cluster(node_key, int(self._clock))
            cow_action = self._make_action(
                action_type="copy_on_write",
                node_id=node_key,
                node_type=node_type,
                old_cluster_id=cow.old_cluster_id,
                new_cluster_id="",
                candidate_type="",
                candidate_id="",
                merge_similarity=0.0,
                cluster_size_before=cow.cluster_size_before,
                cluster_size_after=cow.cluster_size_after,
                physical_state_id_before=old_slot,
                physical_state_id_after="",
                role=role,
                score=score,
            )
            if int(cow.cluster_size_after) <= 0:
                self.state_merger.remove_empty_cluster(int(cow.old_cluster_id))
                slot = old_slot
                self.node_to_slot[node_key] = slot
                self._node_ids[slot] = node_key
                self._type_names[slot] = self._type_key(node_type)
            else:
                self._maybe_degrade_cluster_to_singleton(int(cow.old_cluster_id), role, score)
                slot = self._activate(node_key, self._type_key(node_type), current_node_ids)
            cow_action["physical_state_id_after"] = int(slot)
            return old_state, int(slot), cow_action
        singleton_slot = self.state_merger.singleton_slot_for_node(node_key)
        if singleton_slot is not None:
            slot = int(singleton_slot)
            return self._state_array[slot].astype(np.float32, copy=True), slot, None
        type_key = self._type_key(node_type)
        if type_key == "process" or self.probationary_min_count <= 1:
            slot = self._activate(node_key, type_key, current_node_ids=current_node_ids)
            old_state = np.zeros((self.state_dim,), dtype=np.float32)
            return old_state, int(slot), None
        probationary = self._ensure_probationary(node_key, type_key, count_event=False)
        probationary.count += 1
        probationary.last_seen = int(self._clock)
        role_key = str(role).strip().lower()
        if role_key == "src":
            probationary.as_src_count += 1
        elif role_key == "dst":
            probationary.as_dst_count += 1
        if score is not None and np.isfinite(float(score)):
            score_value = float(score)
            probationary.last_score = score_value
            probationary.recent_score_max = max(float(probationary.recent_score_max), score_value)
        if bool(protected):
            probationary.ever_alerted = True
        old_state = probationary.pending_state.astype(np.float32, copy=True)
        probationary.pending_state = state_model.step(
            probationary.pending_state,
            message,
            q_t=q_t,
        )
        if not self._should_promote(probationary):
            self._evict_probationary_if_needed()
            return old_state, None, None
        slot = self._activate(node_key, type_key, current_node_ids=current_node_ids)
        self._state_array[slot] = probationary.pending_state
        self._probationary.pop(node_key, None)
        self.promoted_nodes_total += 1
        return None, int(slot), None

    def _try_merge_updated_singleton(
        self,
        node_id: int,
        node_type: str,
        slot: int,
        protected: bool,
        role: str,
        score: float | None,
        cow_action: dict[str, Any] | None,
    ) -> None:
        if self.state_merger is None:
            return
        candidate = self._select_merge_candidate(node_id, node_type, slot)
        if candidate is None:
            if cow_action is not None:
                self.state_merger.num_became_singleton_total += 1
                self._append_merge_action(
                    self._make_action(
                        action_type="became_singleton",
                        node_id=node_id,
                        node_type=node_type,
                        old_cluster_id=cow_action.get("old_cluster_id", ""),
                        new_cluster_id="",
                        candidate_type="",
                        candidate_id="",
                        merge_similarity=0.0,
                        cluster_size_before=1,
                        cluster_size_after=1,
                        physical_state_id_before=slot,
                        physical_state_id_after=slot,
                        role=role,
                        score=score,
                    ),
                )
            return
        candidate_type, candidate_id, similarity = candidate
        if candidate_type == "cluster":
            self._merge_singleton_into_cluster(
                node_id,
                node_type,
                slot,
                int(candidate_id),
                float(similarity),
                protected,
                role,
                score,
            )
            return
        self._merge_singleton_with_singleton(
            node_id,
            node_type,
            slot,
            int(candidate_id),
            float(similarity),
            protected,
            role,
            score,
        )

    def _select_merge_candidate(
        self,
        node_id: int,
        node_type: str,
        slot: int,
    ) -> tuple[str, int, float] | None:
        if self.state_merger is None:
            return None
        if str(self.state_merge_config.match_backend) == "bucketed":
            touch_count = int(self._touch_count[int(slot)])
            if touch_count < int(self.state_merge_config.merge_min_count):
                self.state_merger.skipped_merge_due_to_min_count += 1
                return None
            interval = max(int(self.state_merge_config.merge_interval), 1)
            if interval > 1 and touch_count % interval != 0:
                self.state_merger.skipped_merge_due_to_interval += 1
                return None
        scan_started = time.perf_counter()
        current_signature = self.state_merger.signature(self._state_array[slot])
        candidates = self._merge_candidates(
            node_id=node_id,
            node_type=node_type,
            slot=slot,
            current_signature=current_signature,
        )
        self.state_merger.observe_candidate_scan(
            len(candidates),
            time.perf_counter() - scan_started,
        )
        if not candidates:
            return None
        if self.state_merger.mode == "random":
            self.state_merger.num_random_merge_attempts_total += 1
            if self.state_merger.random_value() >= float(self.state_merge_config.random_prob):
                return None
            choice_index = int(self.state_merger.random_value() * len(candidates))
            choice_index = min(choice_index, len(candidates) - 1)
            candidate_type, candidate_id, candidate_signature = candidates[choice_index]
            similarity = float(
                np.dot(self.state_merger.signature(self._state_array[slot]), candidate_signature),
            )
            self.state_merger.num_random_merge_success_total += 1
            self._append_merge_action(
                self._make_action(
                    action_type="random_merge_success",
                    node_id=node_id,
                    node_type=node_type,
                    old_cluster_id="",
                    new_cluster_id="",
                    candidate_type=candidate_type,
                    candidate_id=candidate_id,
                    merge_similarity=similarity,
                    cluster_size_before=1,
                    cluster_size_after=1,
                    physical_state_id_before=slot,
                    physical_state_id_after=slot,
                    role="",
                    score=None,
                ),
            )
            return candidate_type, int(candidate_id), similarity
        self.state_merger.num_merge_attempts_total += 1
        best: tuple[str, int, float] | None = None
        for candidate_type, candidate_id, candidate_signature in candidates:
            similarity = float(np.dot(current_signature, candidate_signature))
            if best is None or similarity > best[2]:
                best = (candidate_type, int(candidate_id), similarity)
        if best is None or best[2] < float(self.state_merge_config.threshold):
            return None
        return best

    def _merge_candidates(
        self,
        *,
        node_id: int,
        node_type: str,
        slot: int,
        current_signature: np.ndarray,
    ) -> list[tuple[str, int, np.ndarray]]:
        if self.state_merger is None:
            return []
        if str(self.state_merge_config.match_backend) != "bucketed":
            return self._same_type_candidates(node_id, node_type, slot)
        refs = self.state_merger.bucketed_candidate_refs(
            node_id=int(node_id),
            node_type=node_type,
            signature=current_signature,
        )
        candidates: list[tuple[str, int, np.ndarray]] = []
        for candidate_type, candidate_id in refs:
            if candidate_type == "cluster":
                cluster = self.state_merger.clusters.get(int(candidate_id))
                if cluster is not None:
                    candidates.append(("cluster", int(candidate_id), cluster.centroid_signature))
                continue
            record = self.state_merger.node_records.get(int(candidate_id))
            if record is None or record.singleton_slot is None:
                continue
            signature = record.singleton_signature
            if signature is None:
                signature = self.state_merger.signature(
                    self._state_array[int(record.singleton_slot)],
                )
            candidates.append(("singleton", int(candidate_id), signature))
        return candidates

    def _same_type_candidates(
        self,
        node_id: int,
        node_type: str,
        slot: int,
    ) -> list[tuple[str, int, np.ndarray]]:
        if self.state_merger is None:
            return []
        type_key = self._type_key(node_type)
        candidates: list[tuple[str, int, np.ndarray]] = []
        for cluster_id, cluster in self.state_merger.clusters.items():
            if str(cluster.cluster_type) != type_key:
                continue
            candidates.append(("cluster", int(cluster_id), cluster.centroid_signature))
        for other_id, record in self.state_merger.node_records.items():
            if int(other_id) == int(node_id):
                continue
            if str(record.node_type) != type_key or record.singleton_slot is None:
                continue
            other_slot = int(record.singleton_slot)
            if other_slot == int(slot):
                continue
            signature = record.singleton_signature
            if signature is None:
                signature = self.state_merger.signature(self._state_array[other_slot])
            candidates.append(
                ("singleton", int(other_id), signature),
            )
        return candidates

    def _merge_singleton_into_cluster(
        self,
        node_id: int,
        node_type: str,
        slot: int,
        cluster_id: int,
        similarity: float,
        protected: bool,
        role: str,
        score: float | None,
    ) -> None:
        if self.state_merger is None:
            return
        cluster = self.state_merger.cluster(cluster_id)
        cluster_slot = int(cluster.centroid_state_slot)
        before = int(cluster.member_count)
        centroid = (
            float(before) * self._state_array[cluster_slot] + self._state_array[int(slot)]
        ) / np.float32(before + 1)
        self._state_array[cluster_slot] = centroid.astype(np.float32, copy=False)
        self.state_merger.join_cluster(
            node_id,
            cluster_id,
            cluster_slot,
            self._state_array[cluster_slot],
            int(self._clock),
            similarity,
        )
        self._release_active_slot(int(slot))
        new_slot = int(self.state_merger.cluster(cluster_id).centroid_state_slot)
        if protected:
            self._protect_until[new_slot] = max(
                int(self._protect_until[new_slot]),
                int(self._clock + self.alert_protect_events),
            )
        self._append_merge_action(
            self._make_action(
                action_type="merge_into_cluster",
                node_id=node_id,
                node_type=node_type,
                old_cluster_id="",
                new_cluster_id=cluster_id,
                candidate_type="cluster",
                candidate_id=cluster_id,
                merge_similarity=similarity,
                cluster_size_before=before,
                cluster_size_after=before + 1,
                physical_state_id_before=slot,
                physical_state_id_after=new_slot,
                role=role,
                score=score,
            ),
        )

    def _merge_singleton_with_singleton(
        self,
        node_id: int,
        node_type: str,
        slot: int,
        candidate_node_id: int,
        similarity: float,
        protected: bool,
        role: str,
        score: float | None,
    ) -> None:
        if self.state_merger is None:
            return
        candidate_slot = self.state_merger.singleton_slot_for_node(candidate_node_id)
        if candidate_slot is None:
            return
        candidate_slot = int(candidate_slot)
        centroid = (self._state_array[int(slot)] + self._state_array[candidate_slot]) / np.float32(2.0)
        self._state_array[int(slot)] = centroid.astype(np.float32, copy=False)
        final_slot = int(slot)
        if candidate_slot != int(slot):
            last_slot = int(self._state_size) - 1
            if int(slot) == last_slot:
                final_slot = candidate_slot
            self._release_active_slot(candidate_slot)
        cluster_id = self.state_merger.create_cluster_from_singletons(
            node_id,
            candidate_node_id,
            final_slot,
            self._state_array[final_slot],
            int(self._clock),
            similarity,
        )
        self.node_to_slot.pop(int(node_id), None)
        self.node_to_slot.pop(int(candidate_node_id), None)
        self._node_ids[final_slot] = int(node_id)
        self._type_names[final_slot] = self._type_key(node_type)
        if protected:
            self._protect_until[final_slot] = max(
                int(self._protect_until[final_slot]),
                int(self._clock + self.alert_protect_events),
            )
        self._append_merge_action(
            self._make_action(
                action_type="merge_with_singleton_create_cluster",
                node_id=node_id,
                node_type=node_type,
                old_cluster_id="",
                new_cluster_id=cluster_id,
                candidate_type="singleton",
                candidate_id=candidate_node_id,
                merge_similarity=similarity,
                cluster_size_before=1,
                cluster_size_after=2,
                physical_state_id_before=slot,
                physical_state_id_after=final_slot,
                role=role,
                score=score,
            ),
        )

    def _maybe_degrade_cluster_to_singleton(
        self,
        cluster_id: int,
        role: str,
        score: float | None,
    ) -> None:
        if self.state_merger is None:
            return
        cluster = self.state_merger.clusters.get(int(cluster_id))
        if cluster is None:
            return
        if int(cluster.member_count) != 1 or not bool(cluster.representative_known):
            return
        representative = int(cluster.representative_node_id)
        slot = int(cluster.centroid_state_slot)
        cluster_type = str(cluster.cluster_type)
        self.state_merger.register_singleton(
            representative,
            cluster_type,
            slot,
            self._state_array[slot],
            int(self._clock),
        )
        self.state_merger.clusters.pop(int(cluster_id), None)
        self.node_to_slot[representative] = slot
        self._node_ids[slot] = representative
        self._type_names[slot] = cluster_type
        self._append_merge_action(
            self._make_action(
                action_type="cluster_dissolved_to_singleton",
                node_id=representative,
                node_type=cluster_type,
                old_cluster_id=cluster_id,
                new_cluster_id="",
                candidate_type="",
                candidate_id="",
                merge_similarity=0.0,
                cluster_size_before=1,
                cluster_size_after=1,
                physical_state_id_before=slot,
                physical_state_id_after=slot,
                role=role,
                score=score,
            ),
        )

    def _append_merge_action(self, action: dict[str, Any]) -> None:
        self.last_merge_actions.append(action)

    def _make_action(
        self,
        action_type: str,
        node_id: int,
        node_type: str,
        old_cluster_id: int | str | None,
        new_cluster_id: int | str | None,
        candidate_type: str,
        candidate_id: int | str | None,
        merge_similarity: float,
        cluster_size_before: int,
        cluster_size_after: int,
        physical_state_id_before: int | str | None,
        physical_state_id_after: int | str | None,
        role: str,
        score: float | None,
    ) -> dict[str, Any]:
        return {
            "event_idx": int(self._clock),
            "action_type": str(action_type),
            "node_id": int(node_id),
            "node_type": self._type_key(node_type),
            "old_cluster_id": "" if old_cluster_id is None else old_cluster_id,
            "new_cluster_id": "" if new_cluster_id is None else new_cluster_id,
            "candidate_type": str(candidate_type),
            "candidate_id": "" if candidate_id is None else candidate_id,
            "merge_mode": str(self.state_merge_config.mode),
            "merge_similarity": float(merge_similarity),
            "merge_threshold": float(self.state_merge_config.threshold),
            "cluster_size_before": int(cluster_size_before),
            "cluster_size_after": int(cluster_size_after),
            "event_score": "" if score is None else float(score),
            "raw_action": "",
            "src_or_dst": str(role),
            "physical_state_id_before": (
                "" if physical_state_id_before is None else physical_state_id_before
            ),
            "physical_state_id_after": (
                "" if physical_state_id_after is None else physical_state_id_after
            ),
        }

    def _release_active_slot(self, slot: int) -> None:
        slot_key = int(slot)
        old_node = int(self._node_ids[slot_key])
        self.node_to_slot.pop(old_node, None)
        last_slot = int(self._state_size) - 1
        if slot_key != last_slot:
            moved_node = int(self._node_ids[last_slot])
            self._state_array[slot_key] = self._state_array[last_slot]
            self._node_ids[slot_key] = self._node_ids[last_slot]
            self._type_names[slot_key] = self._type_names[last_slot]
            self._last_touch[slot_key] = self._last_touch[last_slot]
            self._touch_count[slot_key] = self._touch_count[last_slot]
            self._protect_until[slot_key] = self._protect_until[last_slot]
            if moved_node in self.node_to_slot:
                self.node_to_slot[moved_node] = slot_key
            if self.state_merger is not None:
                self.state_merger.replace_slot(last_slot, slot_key)
        self._state_array[last_slot].fill(0.0)
        self._node_ids[last_slot] = 0
        self._type_names[last_slot] = "unknown"
        self._last_touch[last_slot] = 0
        self._touch_count[last_slot] = 0
        self._protect_until[last_slot] = 0
        self._state_size -= 1

    def _evict_active_if_needed(self, current_node_ids: Iterable[int]) -> None:
        current = {int(node_id) for node_id in current_node_ids}
        while self.active_max_nodes > 0 and self._state_size >= self.active_max_nodes:
            slot = self._find_evictable_slot(current)
            if slot is None:
                return
            self._evict_slot(slot)

    def _find_evictable_slot(self, current_node_ids: set[int]) -> int | None:
        candidates: list[tuple[int, int, int, int]] = []
        for slot in range(self._state_size):
            node_id = int(self._node_ids[slot])
            if node_id in current_node_ids:
                continue
            if self.state_merger is not None and self._cluster_id_for_slot(slot) is not None:
                continue
            if int(self._protect_until[slot]) >= int(self._clock):
                continue
            priority = int(TYPE_PRIORITY.get(self._type_names[slot], 0))
            candidates.append(
                (
                    priority,
                    int(self._last_touch[slot]),
                    int(self._touch_count[slot]),
                    slot,
                ),
            )
        if not candidates:
            return None
        candidates.sort()
        return int(candidates[0][3])

    def _evict_slot(self, slot: int) -> None:
        old_node = int(self._node_ids[slot])
        if self.state_merger is not None:
            self.state_merger.unregister_node(old_node)
        self.node_to_slot.pop(old_node, None)
        last_slot = int(self._state_size) - 1
        if slot != last_slot:
            moved_node = int(self._node_ids[last_slot])
            self._state_array[slot] = self._state_array[last_slot]
            self._node_ids[slot] = self._node_ids[last_slot]
            self._type_names[slot] = self._type_names[last_slot]
            self._last_touch[slot] = self._last_touch[last_slot]
            self._touch_count[slot] = self._touch_count[last_slot]
            self._protect_until[slot] = self._protect_until[last_slot]
            if moved_node in self.node_to_slot:
                self.node_to_slot[moved_node] = slot
            if self.state_merger is not None:
                self.state_merger.replace_slot(last_slot, slot)
        self._state_array[last_slot].fill(0.0)
        self._node_ids[last_slot] = 0
        self._type_names[last_slot] = "unknown"
        self._last_touch[last_slot] = 0
        self._touch_count[last_slot] = 0
        self._protect_until[last_slot] = 0
        self._state_size -= 1
        self.evicted_active_nodes_total += 1

    def _cluster_id_for_slot(self, slot: int) -> int | None:
        if self.state_merger is None:
            return None
        slot_key = int(slot)
        for cluster_id, cluster in self.state_merger.clusters.items():
            if int(cluster.centroid_state_slot) == slot_key:
                return int(cluster_id)
        return None

    def _evict_probationary_if_needed(self) -> None:
        if self.probationary_max_nodes <= 0:
            return
        while len(self._probationary) > self.probationary_max_nodes:
            evict_node = min(
                self._probationary.items(),
                key=lambda item: (int(item[1].last_seen), int(item[1].count), int(item[0])),
            )[0]
            self._probationary.pop(int(evict_node), None)
            self.evicted_probationary_nodes_total += 1

    @staticmethod
    def _grow_1d_array(array: np.ndarray, new_capacity: int, dtype: Any) -> np.ndarray:
        grown = np.zeros((new_capacity,), dtype=dtype)
        if array.size:
            grown[: array.size] = array
        return grown
