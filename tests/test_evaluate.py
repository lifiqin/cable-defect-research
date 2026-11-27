import unittest

from cable_research.evaluate import metrics_summary


class EvaluationTests(unittest.TestCase):
    def test_summary_retains_class_ids_and_distinct_metrics(self):
        class Box:
            ap_class_index = (1, 0)
            ap50, ap, p, r = (0.8, 0.6), (0.4, 0.3), (0.7, 0.5), (0.6, 0.4)
            map50, map = 0.7, 0.35

        result = metrics_summary(Box(), {0: "scratch", 1: "hole"})
        self.assertEqual(result["classes"]["hole"]["ap50"], 0.8)
        self.assertEqual(result["classes"]["scratch"]["ap50_95"], 0.3)
        self.assertEqual(result["map50_95"], 0.35)


if __name__ == "__main__":
    unittest.main()
