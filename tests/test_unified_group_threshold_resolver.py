import unittest

from cs4m.thresholds.unified_group import GroupKey, UnifiedGroupThresholdResolver


class UnifiedGroupThresholdResolverTest(unittest.TestCase):
    def test_validation_group_quantile_wins_when_supported(self):
        key = GroupKey("event_semantic_target", "EVENT_READ", "file", "process")
        resolver = UnifiedGroupThresholdResolver(
            validation_groups={
                key: {
                    "validation_count": 120,
                    "validation_quantile_threshold": 0.7,
                }
            },
            train_groups={key: {"train_count": 100, "train_max_score": 0.5}},
            validation_global_threshold=0.9,
            min_validation_count=100,
            min_train_count=20,
        )

        resolved = resolver.resolve(key)

        self.assertEqual(resolved.threshold_source, "validation_group_quantile")
        self.assertEqual(resolved.threshold, 0.7)

    def test_train_max_fallback_when_validation_missing(self):
        key = GroupKey("event_semantic_target", "EVENT_CONNECT", "process", "netflow")
        resolver = UnifiedGroupThresholdResolver(
            validation_groups={},
            train_groups={key: {"train_count": 21, "train_max_score": 0.4}},
            validation_global_threshold=0.9,
            min_validation_count=100,
            min_train_count=20,
            train_margin_abs=0.005,
            train_margin_rel=0.05,
        )

        resolved = resolver.resolve(key)

        self.assertEqual(resolved.threshold_source, "train_group_max")
        self.assertAlmostEqual(resolved.threshold, 0.42)

    def test_global_fallback_for_cold_group(self):
        key = GroupKey("event_semantic_target", "EVENT_EXECUTE", "process", "file")
        resolver = UnifiedGroupThresholdResolver(
            validation_groups={},
            train_groups={},
            validation_global_threshold=0.9,
        )

        resolved = resolver.resolve(key)

        self.assertEqual(resolved.threshold_source, "validation_global_quantile")
        self.assertEqual(resolved.threshold, 0.9)


if __name__ == "__main__":
    unittest.main()
