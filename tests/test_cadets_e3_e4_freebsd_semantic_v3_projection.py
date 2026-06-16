import unittest

from scripts.tools.project_cadets_e3_e4_freebsd_semantic_v3_candidates import (
    ProjectionAccumulator,
    candidate_file_tokens,
    candidate_netflow_tokens,
    candidate_process_tokens,
    render_projection_report,
    token_entropy_risk,
)


class CadetsE3E4FreeBsdSemanticV3ProjectionTests(unittest.TestCase):
    def test_candidate_file_splits_empty_file_bucket(self):
        self.assertEqual(
            candidate_file_tokens(""),
            ("file", "missing_file_path", "missing_path"),
        )

    def test_candidate_process_refines_process_other_by_shape(self):
        tokens = candidate_process_tokens("/tmp/abc123x")
        self.assertEqual(tokens[0], "process")
        self.assertEqual(tokens[1], "process_other_payload_shape")
        self.assertEqual(tokens[2], "abc123x")

    def test_candidate_file_does_not_treat_home_lock_as_payload_shape(self):
        tokens = candidate_file_tokens("/home/george/Sent.lock")
        self.assertEqual(tokens, ("file", "user_home_file", "sent_lock"))

    def test_candidate_file_classifies_home_web_artifacts_separately(self):
        tokens = candidate_file_tokens("/usr/home/user/eraseme/index.html")
        self.assertEqual(tokens, ("file", "user_home_web_artifact_file", "eraseme_index_html"))

    def test_candidate_netflow_preserves_current_exact_ip(self):
        tokens = candidate_netflow_tokens("8.8.8.8", "128.55.12.10")
        self.assertIn("8_8_8_8", tokens)

    def test_projection_accumulator_tracks_current_candidate_deltas(self):
        acc = ProjectionAccumulator(dataset="CADETS_E3", gt_nodes={1})
        acc.add_node(
            node_id=1,
            kind="file",
            current_tokens=("file",),
            candidate_tokens=("file", "missing_file_path", "missing_path"),
            raw_detail="",
        )
        row = next(row for row in acc.summary_rows() if row["kind"] == "file")
        self.assertEqual(row["current_empty_file_nodes"], 1)
        self.assertEqual(row["candidate_empty_file_nodes"], 0)
        self.assertEqual(row["gt_current_generic_nodes"], 1)
        self.assertEqual(row["gt_candidate_generic_nodes"], 1)

    def test_token_entropy_risk_flags_long_or_hex_tokens(self):
        self.assertIn("long_token", token_entropy_risk(("process", "x" * 90)))
        self.assertIn("hex_like_token", token_entropy_risk(("file", "a1b2c3d4e5f6")))

    def test_report_mentions_safety_constraints(self):
        report = render_projection_report(
            output_dir="outputs/diagnostics/example",
            summary_rows=[],
            rule_rows=[],
            missing_mapping_count=0,
            prior_audit_dir="outputs/diagnostics/audit",
        )
        self.assertIn("No Word2Vec training", report)
        self.assertIn("No inference", report)
        self.assertIn("GT is post-hoc only", report)
        self.assertIn("official tokenizer is not modified", report)
        self.assertIn("Recommended next stage", report)


if __name__ == "__main__":
    unittest.main()
