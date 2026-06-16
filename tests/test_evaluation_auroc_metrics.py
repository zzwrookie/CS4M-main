import unittest

from cs4m.evaluation.auroc import compute_auroc, roc_curve_points


class EvaluationAurocMetricsTest(unittest.TestCase):
    def test_compute_auroc_returns_expected_pairwise_value(self):
        value = compute_auroc(
            [
                {"id": "a", "score": 0.9, "y_true": 1},
                {"id": "b", "score": 0.8, "y_true": 0},
                {"id": "c", "score": 0.2, "y_true": 1},
                {"id": "d", "score": 0.1, "y_true": 0},
            ]
        )

        self.assertAlmostEqual(value, 0.75)

    def test_compute_auroc_returns_none_for_single_class(self):
        value = compute_auroc(
            [
                {"id": "a", "score": 0.9, "y_true": 1},
                {"id": "b", "score": 0.8, "y_true": 1},
            ]
        )

        self.assertIsNone(value)

    def test_roc_curve_points_are_threshold_fpr_tpr_rows(self):
        rows = roc_curve_points(
            [
                {"id": "a", "score": 0.9, "y_true": 1},
                {"id": "b", "score": 0.8, "y_true": 0},
                {"id": "c", "score": 0.2, "y_true": 1},
            ]
        )

        self.assertEqual(list(rows[0]), ["threshold", "fpr", "tpr"])
        self.assertEqual(rows[0]["threshold"], "inf")
        self.assertAlmostEqual(rows[-1]["tpr"], 1.0)


if __name__ == "__main__":
    unittest.main()
