import unittest

from scripts.tools.audit_optc201_gt_split_alignment import split_presence_status


class Optc201GtSplitAlignmentTests(unittest.TestCase):
    def test_split_presence_status_validation_only(self) -> None:
        self.assertEqual(
            split_presence_status({"train": 0, "validation": 5, "test": 0}),
            "validation_only",
        )

    def test_split_presence_status_test_present(self) -> None:
        self.assertEqual(
            split_presence_status({"train": 0, "validation": 0, "test": 2}),
            "test_present",
        )

    def test_split_presence_status_mixed_with_test(self) -> None:
        self.assertEqual(
            split_presence_status({"train": 1, "validation": 0, "test": 2}),
            "mixed_with_test",
        )

    def test_split_presence_status_not_in_event_index(self) -> None:
        self.assertEqual(
            split_presence_status({"train": 0, "validation": 0, "test": 0}),
            "not_in_event_index",
        )


if __name__ == "__main__":
    unittest.main()
