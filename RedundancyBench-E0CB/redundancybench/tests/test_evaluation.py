"""Small scoring examples that distinguish the three task denominators."""

import json
import unittest

from redundancybench.evaluation import evaluate_payload
from redundancybench.LLM_judge.response_parser import parse_detect


STEP_A = [2, 3]
STEP_B = [6, 7]


def payload(task, rows):
    return {"metadata": {"task": task, "requested_units": len(rows)}, "results": rows}


class EvaluateTests(unittest.TestCase):
    def test_step_classification_uses_all_gold_cases(self):
        rows = [
            {"identity": "a", "domain": "airline", "dataset_id": "human_annotation", "redundant_step_type": "Duplicated Step", "predicted_type": "Duplicated Step"},
            {"identity": "b", "domain": "airline", "dataset_id": "human_annotation", "redundant_step_type": "Incorrect Step", "predicted_type": None, "format_failed": True},
        ]
        self.assertEqual(evaluate_payload(payload("classify", rows))["overall"]["accuracy"], 0.5)

    def test_evidence_requires_both_type_and_exact_evidence(self):
        rows = [
            {"identity": "a", "domain": "retail", "dataset_id": "human_annotation", "redundant_step_type": "Duplicated Step", "evidence_step_idx": [2, 3], "predicted_type": "Duplicated Step", "predicted_evidence_idx": [2, 3]},
            {"identity": "b", "domain": "retail", "dataset_id": "human_annotation", "redundant_step_type": "Incorrect Step", "evidence_step_idx": [-1], "predicted_type": "Incorrect Step", "predicted_evidence_idx": [2, 3]},
        ]
        report = evaluate_payload(payload("retrieve", rows))
        self.assertEqual(report["overall"], {"gold_count": 2, "correct": 1, "accuracy": 0.5})
        self.assertTrue(report["per_unit"][1]["type_correct"])

    def test_detection_ranks_confidence_and_counts_unique_exact_triples(self):
        gold_a = {"redundant_step_idx": STEP_A, "redundant_step_type": "Duplicated Step", "evidence_step_idx": [0, 1]}
        gold_b = {"redundant_step_idx": STEP_B, "redundant_step_type": "Incorrect Step", "evidence_step_idx": [-1]}
        raw = [
            {**gold_a, "confidence": 0.8},
            {**gold_a, "confidence": 0.9},
            {**gold_b, "confidence": 0.7},
            {**gold_b, "redundant_step_type": "Exploratory Step", "confidence": 0.6},
        ]
        parsed = parse_detect(json.dumps(raw))
        rows = [
            {"identity": "a", "domain": "telecom", "dataset_id": "automated_annotation", "gold": [gold_a, gold_b], "predictions": parsed["predictions"]},
            {"identity": "b", "domain": "telecom", "dataset_id": "automated_annotation", "gold": [gold_a], "predictions": []},
        ]
        report = evaluate_payload(payload("detect", rows))
        self.assertEqual(report["per_unit"][0]["hits"], {"1": 1, "5": 2, "10": 2})
        self.assertEqual(report["per_unit"][0]["p_at_k"], {"1": 1.0, "5": 0.5, "10": 0.5})
        self.assertEqual(report["overall"]["p_at_k"], {"1": 0.5, "5": 0.25, "10": 0.25})
        self.assertAlmostEqual(report["overall"]["avg_p_at_k"], 1 / 3)

    def test_incomplete_run_does_not_claim_a_score(self):
        with self.assertRaises(ValueError):
            evaluate_payload({"metadata": {"task": "classify", "requested_units": 2}, "results": []})

    def test_later_prediction_for_same_step_cannot_score(self):
        gold = {"redundant_step_idx": STEP_A, "redundant_step_type": "Duplicated Step", "evidence_step_idx": [0, 1]}
        predictions = parse_detect(json.dumps([
            {**gold, "evidence_step_idx": [4, 5], "confidence": 0.9},
            {**gold, "confidence": 0.8},
        ]))["predictions"]
        self.assertTrue(predictions[1]["duplicate"])
        self.assertFalse(predictions[1]["valid"])
        row = {"identity": "a", "domain": "airline", "dataset_id": "human_annotation", "gold": [gold], "predictions": predictions}
        report = evaluate_payload(payload("detect", [row]))
        self.assertEqual(report["overall"]["p_at_k"]["1"], 0.0)
        self.assertEqual(report["overall"]["p_at_k"]["5"], 0.0)

    def test_invalid_confidence_occupies_first_penalty_slot(self):
        gold = {"redundant_step_idx": STEP_A, "redundant_step_type": "Duplicated Step", "evidence_step_idx": [0, 1]}
        predictions = parse_detect(json.dumps([
            {**gold, "confidence": 0.8},
            {**gold, "confidence": "unknown"},
        ]))["predictions"]
        self.assertEqual([item["source_rank"] for item in predictions], [2, 1])
        self.assertIsNone(predictions[0]["confidence"])
        row = {"identity": "a", "domain": "airline", "dataset_id": "human_annotation", "gold": [gold], "predictions": predictions}
        report = evaluate_payload(payload("detect", [row]))
        self.assertEqual(report["overall"]["p_at_k"]["1"], 0.0)
        self.assertEqual(report["overall"]["p_at_k"]["5"], 0.5)


if __name__ == "__main__":
    unittest.main()
