import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import pandas as pd

import analyze_results


class AnalyzeResultsTests(unittest.TestCase):
    def test_main_reports_category_and_density_metrics(self):
        rows = [
            {
                "unique_id": 1,
                "Dax_or_Human": "Human",
                "Possible_Biased_Terms": '["obese"]',
                "Likely_Biased_Terms": '["difficult patient"]',
                "Possible_Biased_Terms_Normalized": '["obese"]',
                "Likely_Biased_Terms_Normalized": '["difficult patient"]',
                "Possible_Bias_Categories": '["weight-based identity label"]',
                "Likely_Bias_Categories": '["difficult-patient framing"]',
                "Possible_Bias_Count": 1,
                "Likely_Bias_Count": 1,
                "Note_Word_Count": 100,
                "Prompt_Version": "test-prompt",
                "Pipeline_Version": "test-pipeline",
                "Model_Used": "azure-test",
            },
            {
                "unique_id": 2,
                "Dax_or_Human": "Dax",
                "Possible_Biased_Terms": "[]",
                "Likely_Biased_Terms": "[]",
                "Possible_Biased_Terms_Normalized": "[]",
                "Likely_Biased_Terms_Normalized": "[]",
                "Possible_Bias_Categories": "[]",
                "Likely_Bias_Categories": "[]",
                "Possible_Bias_Count": 0,
                "Likely_Bias_Count": 0,
                "Note_Word_Count": 200,
                "Prompt_Version": "test-prompt",
                "Pipeline_Version": "test-pipeline",
                "Model_Used": "azure-test",
            },
        ]

        with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=True) as handle:
            pd.DataFrame(rows).to_csv(handle.name, index=False)
            original_argv = sys.argv
            sys.argv = ["analyze_results.py", handle.name, "--top-n", "3"]
            try:
                output = io.StringIO()
                with redirect_stdout(output):
                    analyze_results.main()
            finally:
                sys.argv = original_argv

        text = output.getvalue()
        self.assertIn("Likely flags / 1k words", text)
        self.assertIn("Top 3 likely categories", text)
        self.assertIn("difficult-patient framing", text)
        self.assertIn("Human likely category note prevalence", text)


if __name__ == "__main__":
    unittest.main()
