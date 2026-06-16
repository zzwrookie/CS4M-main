import unittest

from scripts.tools.audit_cadets_e3_e4_freebsd_semantics import (
    AuditAccumulator,
    build_candidate_design_rows,
    classify_focus_flags,
    render_markdown_report,
    token_is_generic,
)


class CadetsE3E4FreeBsdSemanticAuditTests(unittest.TestCase):
    def test_accumulator_tracks_generic_and_gt_posthoc(self):
        acc = AuditAccumulator(dataset="CADETS_E3", gt_nodes={10})

        acc.add_node(
            node_id=10,
            kind="process",
            tokens=("process", "process_other", "vugefal"),
            raw_detail="vugefal",
        )
        acc.add_node(
            node_id=11,
            kind="process",
            tokens=("process", "ssh_service", "sshd"),
            raw_detail="sshd",
        )

        rows = acc.summary_rows()
        process = next(row for row in rows if row["kind"] == "process")
        self.assertEqual(process["total_nodes"], 2)
        self.assertEqual(process["generic_nodes"], 1)
        self.assertEqual(process["gt_nodes"], 1)
        self.assertEqual(process["gt_generic_nodes"], 1)

    def test_focus_flags_identify_payload_and_empty_file(self):
        self.assertIn(
            "process_other",
            classify_focus_flags("process", ("process", "process_other", "xim"), "xim"),
        )
        self.assertIn("payload_like_detail", classify_focus_flags("file", ("file",), ""))
        self.assertIn(
            "device_random",
            classify_focus_flags("file", ("file", "device_file", "random"), "/dev/random"),
        )

    def test_type_prefix_alone_is_not_generic_when_detail_exists(self):
        self.assertFalse(token_is_generic("file", ("file", "tmp_file", "vugefal")))
        self.assertTrue(token_is_generic("file", ("file",)))
        self.assertFalse(
            token_is_generic("netflow", ("netflow", "ip_public_or_external", "1_2_3_4"))
        )
        self.assertTrue(token_is_generic("netflow", ("netflow",)))

    def test_candidate_design_rows_are_label_free(self):
        rows = build_candidate_design_rows()
        names = {row["candidate_name"] for row in rows}
        self.assertIn("process_other_safe_lexical_detail", names)
        self.assertIn("file_empty_bucket_refinement", names)
        for row in rows:
            self.assertEqual(row["uses_gt_for_rule"], False)
            self.assertEqual(row["runtime_visible"], True)

    def test_report_mentions_no_training_or_inference(self):
        acc = AuditAccumulator(dataset="CADETS_E3", gt_nodes={})
        report = render_markdown_report(
            output_dir="outputs/diagnostics/example",
            summary_rows=acc.summary_rows(),
            candidate_rows=build_candidate_design_rows(),
            missing_mapping_count=0,
            e4_artifact_dir="tmp/example",
        )
        self.assertIn("No Word2Vec training", report)
        self.assertIn("No inference", report)
        self.assertIn("GT is post-hoc only", report)


if __name__ == "__main__":
    unittest.main()
