import unittest

from bias_review_utils import enrich_bias_result, infer_bias_category, normalize_term


class BiasReviewUtilsTests(unittest.TestCase):
    def test_enrich_bias_result_filters_known_false_positives(self):
        note = (
            "Chief Complaint: follow-up. The patient denies chest pain. "
            "This is a difficult patient. Airway exam noted difficult airway. "
            "The patient is morbidly obese."
        )
        result = {
            "possible": ["Chief Complaint", "denies", "obese"],
            "likely": ["difficult patient", "difficult airway", "morbidly obese"],
        }

        enriched = enrich_bias_result(note, result)

        self.assertEqual(enriched["possible_terms"], ["obese"])
        self.assertEqual(enriched["likely_terms"], ["difficult patient", "morbidly obese"])
        self.assertEqual(enriched["likely_normalized_terms"], ["difficult patient", "morbid obesity"])
        self.assertIn("The patient is morbidly obese.", [d["context"] for d in enriched["likely_details"]])

    def test_normalize_term_and_category_mapping(self):
        self.assertEqual(normalize_term(" Morbidly Obese "), "morbid obesity")
        self.assertEqual(infer_bias_category("drug-seeking"), "difficult-patient framing")
        self.assertEqual(infer_bias_category("current smoker"), "substance identity label")


if __name__ == "__main__":
    unittest.main()
