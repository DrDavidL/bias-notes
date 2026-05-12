import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

import export_reviewer_adjudication


class ExportReviewerAdjudicationTests(unittest.TestCase):
    def test_script_writes_long_format_csv(self):
        rows = [
            {
                "unique_id": 1,
                "Likely_Bias_Details": '[{"term":"difficult patient","normalized_term":"difficult patient","category":"difficult-patient framing","context":"This is a difficult patient."}]',
                "Possible_Bias_Details": "[]",
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "results.csv"
            output_path = Path(temp_dir) / "reviewer.csv"
            pd.DataFrame(rows).to_csv(input_path, index=False)

            previous_argv = sys.argv
            sys.argv = [
                "export_reviewer_adjudication.py",
                str(input_path),
                "--output",
                str(output_path),
            ]
            try:
                export_reviewer_adjudication.main()
            finally:
                sys.argv = previous_argv

            reviewer_df = pd.read_csv(output_path)

        self.assertEqual(len(reviewer_df), 1)
        self.assertEqual(reviewer_df.loc[0, "model_term"], "difficult patient")


if __name__ == "__main__":
    unittest.main()
