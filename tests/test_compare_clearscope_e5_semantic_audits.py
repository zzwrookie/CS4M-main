from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.tools.compare_clearscope_e5_semantic_audits import compare_audit_dirs


class CompareClearScopeE5SemanticAuditsTests(unittest.TestCase):
    """Validate ClearScope E5 v31/v33 semantic audit comparison."""

    def test_compare_audit_dirs_reports_core_deltas(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            v31 = root / "v31"
            v33 = root / "v33"
            v31.mkdir()
            v33.mkdir()
            (v31 / "label_free_summary.json").write_text(
                json.dumps(
                    {
                        "node_count": 10,
                        "endpoint_node_count": 2,
                        "fallback_counts": {"file_other": 3, "dev_other": 1},
                        "event_tuple_summary": {
                            "input_event_count": 100,
                            "event_count": 100,
                            "skipped_event_count": 0,
                            "usable_event_rate": 1.0,
                            "train_tuple_count": 5,
                            "val_tuple_count": 4,
                            "val_oov_tuple_count": 2,
                            "test_tuple_count": 6,
                            "test_oov_tuple_count": 3,
                            "test_seen_tuple_count": 3,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (v33 / "label_free_summary.json").write_text(
                json.dumps(
                    {
                        "node_count": 10,
                        "endpoint_node_count": 2,
                        "fallback_counts": {"file_other": 1, "dev_other": 1},
                        "event_tuple_summary": {
                            "input_event_count": 100,
                            "event_count": 100,
                            "skipped_event_count": 0,
                            "usable_event_rate": 1.0,
                            "train_tuple_count": 5,
                            "val_tuple_count": 4,
                            "val_oov_tuple_count": 2,
                            "test_tuple_count": 7,
                            "test_oov_tuple_count": 4,
                            "test_seen_tuple_count": 3,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (v31 / "label_aware_diagnostics.json").write_text(
                json.dumps(
                    {
                        "malicious_fallback_counts": {"file_other": 3},
                        "malicious_detail_collision_groups": {"other": ["/a", "/b"]},
                    }
                ),
                encoding="utf-8",
            )
            (v33 / "label_aware_diagnostics.json").write_text(
                json.dumps(
                    {
                        "malicious_fallback_counts": {},
                        "malicious_detail_collision_groups": {},
                    }
                ),
                encoding="utf-8",
            )

            comparison = compare_audit_dirs(v31, v33)

        self.assertEqual(comparison["fallback_counts"]["file_other"]["delta"], -2.0)
        self.assertEqual(comparison["malicious_fallback_counts"]["file_other"]["v33"], 0)
        self.assertEqual(
            comparison["event_tuple_summary"]["test_oov_tuple_count"]["delta"],
            1.0,
        )
        self.assertEqual(comparison["malicious_collision_group_count"]["delta"], -1.0)
        self.assertFalse(comparison["label_free_summary_has_label_fields"])


if __name__ == "__main__":
    unittest.main()
