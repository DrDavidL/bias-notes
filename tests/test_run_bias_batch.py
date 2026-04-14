import unittest

import pandas as pd

from run_bias_batch import select_rows


class RunBiasBatchTests(unittest.TestCase):
    def test_select_rows_is_seeded_for_reproducible_random_samples(self):
        df = pd.DataFrame({"unique_id": list(range(20)), "note_text": [f"note {i}" for i in range(20)]})

        first, first_mode = select_rows(df, charts_to_process=5, selection="random", seed=42)
        second, second_mode = select_rows(df, charts_to_process=5, selection="random", seed=42)

        self.assertEqual(first_mode, "random sample (seed=42)")
        self.assertEqual(second_mode, "random sample (seed=42)")
        self.assertEqual(first["unique_id"].tolist(), second["unique_id"].tolist())

    def test_select_rows_can_take_first_n_rows(self):
        df = pd.DataFrame({"unique_id": list(range(10)), "note_text": [f"note {i}" for i in range(10)]})

        selected, selection_mode = select_rows(df, charts_to_process=3, selection="head", seed=42)

        self.assertEqual(selection_mode, "first N rows")
        self.assertEqual(selected["unique_id"].tolist(), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
