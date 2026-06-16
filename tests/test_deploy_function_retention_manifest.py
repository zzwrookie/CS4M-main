import csv
import tempfile
import unittest
from pathlib import Path

from scripts.tools.build_deploy_function_retention_manifest import (
    DefinitionRecord,
    build_retention_manifest,
    parse_python_definitions,
    write_manifest_outputs,
)


class DeployFunctionRetentionManifestTest(unittest.TestCase):
    def test_parse_python_definitions_finds_functions_classes_and_constants(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.py"
            path.write_text(
                "\n".join(
                    [
                        "CONSTANT = 1",
                        "",
                        "class KeptClass:",
                        "    pass",
                        "",
                        "def kept_function():",
                        "    return CONSTANT",
                    ]
                ),
                encoding="utf-8",
            )

            records = parse_python_definitions(path)

        names = {(record.kind, record.name) for record in records}
        self.assertIn(("constant", "CONSTANT"), names)
        self.assertIn(("class", "KeptClass"), names)
        self.assertIn(("function", "kept_function"), names)

    def test_build_manifest_marks_registry_reachable_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = root / "policy.py"
            module.write_text(
                "\n".join(
                    [
                        "ACTION_TYPE_ALERT_POLICIES = {'cadets': cadets_policy}",
                        "",
                        "def cadets_policy(row):",
                        "    return True",
                        "",
                        "def old_policy(row):",
                        "    return False",
                    ]
                ),
                encoding="utf-8",
            )

            records = build_retention_manifest(
                repo_root=root,
                python_paths=[module],
                explicit_keep_names={"cadets_policy"},
                entrypoint_paths=[module],
            )

        by_name = {record.name: record for record in records}
        self.assertEqual(by_name["cadets_policy"].classification, "keep_runtime")
        self.assertEqual(by_name["old_policy"].classification, "exclude_unused")

    def test_write_manifest_outputs_writes_markdown_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            records = [
                DefinitionRecord(
                    module_path="policy.py",
                    name="cadets_policy",
                    kind="function",
                    line_number=3,
                    classification="keep_runtime",
                    caller_or_reason="explicit_keep",
                    streaming_required=True,
                    touches_ground_truth=False,
                    evaluation_only=False,
                    safe_to_remove=False,
                    notes="policy registry",
                )
            ]

            write_manifest_outputs(records, out_dir)

            csv_path = out_dir / "cs4m_deploy_function_retention_manifest.csv"
            md_path = out_dir / "cs4m_deploy_function_retention_manifest.md"
            self.assertTrue(csv_path.exists())
            self.assertTrue(md_path.exists())
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["name"], "cadets_policy")
            self.assertIn("cadets_policy", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
