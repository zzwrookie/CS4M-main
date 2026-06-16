"""Tests for SSPM stream-batch profiling callbacks."""

from __future__ import annotations

import unittest

import numpy as np

from cs4m.models.cs4m_lowrank import SSPMLowRankConfig, SSPMLowRankModel


class SSPMTrainStreamProfileTests(unittest.TestCase):
    """Validate optional batch profiling without changing default training behavior."""

    def test_train_stream_batches_reports_stack_and_step_seconds(self) -> None:
        model = SSPMLowRankModel(
            SSPMLowRankConfig(
                latent_dim=4,
                target_dim=4,
                state_dim=4,
                rank=2,
                context_mode="with_action",
                global_context_mode="no_global",
            ),
        )
        rows = [
            (np.zeros((model.context_dim,), dtype=np.float32), np.ones((4,), dtype=np.float32)),
            (np.ones((model.context_dim,), dtype=np.float32), np.ones((4,), dtype=np.float32)),
        ]
        profile_rows: list[dict[str, float]] = []

        stats = model.train_stream_batches(
            lambda _epoch: iter(rows),
            epochs=1,
            batch_events=2,
            profile_callback=profile_rows.append,
        )

        self.assertEqual(stats["train_batches_total"], 1)
        self.assertEqual(len(profile_rows), 1)
        self.assertGreaterEqual(profile_rows[0]["batch_stack_seconds"], 0.0)
        self.assertGreaterEqual(profile_rows[0]["train_batch_step_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
