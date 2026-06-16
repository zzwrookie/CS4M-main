import unittest

from cs4m.evaluation.score_samples import derive_node_scores_from_event_rows


class ScoreSampleDerivationTest(unittest.TestCase):
    def test_derive_node_scores_uses_max_event_score_for_each_node(self):
        rows = [
            {
                "event_id": "e1",
                "src_idx": "1",
                "dst_idx": "2",
                "info_src": "1",
                "info_dst": "2",
                "score": "0.3",
            },
            {
                "event_id": "e2",
                "src_idx": "2",
                "dst_idx": "3",
                "info_src": "2",
                "info_dst": "3",
                "score": "0.8",
            },
        ]

        scores = derive_node_scores_from_event_rows(rows)

        self.assertEqual(scores["1"], 0.3)
        self.assertEqual(scores["2"], 0.8)
        self.assertEqual(scores["3"], 0.8)


if __name__ == "__main__":
    unittest.main()
