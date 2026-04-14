import unittest

from bias_review_utils import (
    canonicalize_bias_categories,
    enrich_bias_result,
    infer_bias_category,
    normalize_term,
)


class BiasReviewUtilsTests(unittest.TestCase):
    def test_enrich_bias_result_filters_known_false_positives(self):
        note = (
            "Chief Complaint: follow-up. The patient denies chest pain. "
            "This is a difficult patient. Airway exam noted difficult airway. "
            "The patient is morbidly obese."
        )
        result = {
            "possible": ["Chief Complaint", "denies", {"term": "obese", "categories": ["weight-based identity label"]}],
            "likely": [
                {"term": "difficult patient", "categories": ["difficult-patient framing"]},
                "difficult airway",
                {"term": "morbidly obese", "categories": ["weight-based identity label"]},
            ],
        }

        enriched = enrich_bias_result(note, result)

        self.assertEqual(enriched["possible_terms"], ["obese"])
        self.assertEqual(enriched["likely_terms"], ["difficult patient", "morbidly obese"])
        self.assertEqual(enriched["likely_normalized_terms"], ["difficult patient", "morbid obesity"])
        self.assertIn("The patient is morbidly obese.", [d["context"] for d in enriched["likely_details"]])
        self.assertEqual(
            enriched["likely_details"][0]["categories"],
            ["difficult-patient framing"],
        )

    def test_normalize_term_and_category_mapping(self):
        self.assertEqual(normalize_term(" Morbidly Obese "), "morbid obesity")
        self.assertEqual(infer_bias_category("drug-seeking"), "difficult-patient framing")
        self.assertEqual(infer_bias_category("current smoker"), "substance identity label")
        self.assertEqual(
            canonicalize_bias_categories(["Weight-Based Identity Labels", "difficult patient framing", "unknown label"]),
            ["weight-based identity label", "difficult-patient framing"],
        )

    def test_enrich_bias_result_falls_back_when_model_category_is_invalid(self):
        note = "Patient is obese and was told to lose weight."
        result = {
            "possible": [{"term": "obese", "categories": ["not a real label"]}],
            "likely": [],
        }

        enriched = enrich_bias_result(note, result)

        self.assertEqual(enriched["possible_categories"], ["weight-based identity label"])
        self.assertEqual(enriched["possible_details"][0]["category"], "weight-based identity label")


if __name__ == "__main__":
    unittest.main()
