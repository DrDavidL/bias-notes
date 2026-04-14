import tempfile
import unittest

import pandas as pd
from openai import AzureOpenAI

from bias_pipeline import (
    AzureBiasPipeline,
    AzureBiasPipelineConfig,
    build_analysis_ready_dataframe,
    build_reviewer_adjudication_dataframe,
    chunk_by_sentences,
    postprocess_results,
    write_output_bundle,
)


class BiasPipelineTests(unittest.TestCase):
    class StubPipeline(AzureBiasPipeline):
        def __init__(self, cache_path=None):
            super().__init__(
                AzureOpenAI(
                    api_version="2024-02-15-preview",
                    azure_endpoint="https://example.openai.azure.com/",
                    api_key="test-key",
                ),
                AzureBiasPipelineConfig(
                    prompt_path="bias_detection_prompt.py",
                    model_for_api="test-model",
                    cache_path=cache_path,
                    enable_second_pass_adjudication=False,
                ),
            )
            self.analyze_calls = 0

        def analyze_note_text(self, full_text: str):
            self.analyze_calls += 1
            return {"possible": ["obese"], "likely": ["difficult patient"]}

    def test_chunk_by_sentences_respects_boundaries(self):
        text = "Sentence one. Sentence two is a bit longer. Sentence three."
        chunks = chunk_by_sentences(text, max_chars=30)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(len(chunk) <= 30 for chunk in chunks))
        self.assertEqual(" ".join(chunks), text)

    def test_postprocess_results_filters_physiologic_normal_language(self):
        result = {
            "possible": ["normal", "normal breath sounds", "obese"],
            "likely": ["difficult patient", "mood and affect congruent"],
        }

        filtered = postprocess_results(result)

        self.assertEqual(filtered["possible"], ["obese"])
        self.assertEqual(filtered["likely"], ["difficult patient"])

    def test_analysis_ready_output_drops_note_text_and_details(self):
        df = pd.DataFrame(
            [
                {
                    "unique_id": 1,
                    "note_text": "Example note",
                    "Possible_Bias_Details": '[{"term":"obese"}]',
                    "Likely_Bias_Details": '[{"term":"difficult patient"}]',
                    "Likely_Biased_Terms": '["difficult patient"]',
                }
            ]
        )

        analysis_ready = build_analysis_ready_dataframe(df)

        self.assertNotIn("note_text", analysis_ready.columns)
        self.assertNotIn("Possible_Bias_Details", analysis_ready.columns)
        self.assertNotIn("Likely_Bias_Details", analysis_ready.columns)
        self.assertIn("Likely_Biased_Terms", analysis_ready.columns)

    def test_write_output_bundle_creates_analysis_ready_companion(self):
        df = pd.DataFrame(
            [
                {
                    "unique_id": 1,
                    "note_text": "Example note",
                    "Likely_Biased_Terms": '["difficult patient"]',
                    "Likely_Bias_Details": '[{"term":"difficult patient","normalized_term":"difficult patient","category":"difficult-patient framing","context":"This is a difficult patient."}]',
                }
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            reviewer_path = f"{temp_dir}/results.csv"
            output_paths = write_output_bundle(df, reviewer_path)

            reviewer_df = pd.read_csv(output_paths["reviewer_path"])
            analysis_df = pd.read_csv(output_paths["analysis_path"])
            adjudication_df = pd.read_csv(output_paths["adjudication_path"])

        self.assertIn("note_text", reviewer_df.columns)
        self.assertNotIn("note_text", analysis_df.columns)
        self.assertIn("model_term", adjudication_df.columns)
        self.assertEqual(adjudication_df.loc[0, "model_term"], "difficult patient")

    def test_build_reviewer_adjudication_dataframe_flattens_details(self):
        df = pd.DataFrame(
            [
                {
                    "unique_id": 7,
                    "Dax_or_Human": "Human",
                    "Note_Hash": "abc123",
                    "Prompt_Version": "prompt-v1",
                    "Pipeline_Version": "pipeline-v1",
                    "Model_Used": "azure-test",
                    "Possible_Bias_Details": '[{"term":"obese","normalized_term":"obese","category":"weight-based identity label","context":"Patient is obese."}]',
                    "Likely_Bias_Details": '[{"term":"difficult patient","normalized_term":"difficult patient","category":"difficult-patient framing","context":"This is a difficult patient."}]',
                }
            ]
        )

        adjudication_df = build_reviewer_adjudication_dataframe(df)

        self.assertEqual(len(adjudication_df), 2)
        self.assertEqual(set(adjudication_df["model_bucket"]), {"likely", "possible"})
        self.assertIn("term_context", adjudication_df.columns)

    def test_process_dataframe_reuses_persistent_note_cache(self):
        df = pd.DataFrame([{"unique_id": 1, "note_text": "Same note reused"}])

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = f"{temp_dir}/persistent_cache.csv"

            first_pipeline = self.StubPipeline(cache_path=cache_path)
            first_result = first_pipeline.process_dataframe(df)

            second_pipeline = self.StubPipeline(cache_path=cache_path)
            second_result = second_pipeline.process_dataframe(df)

        self.assertEqual(first_pipeline.analyze_calls, 1)
        self.assertEqual(second_pipeline.analyze_calls, 0)
        self.assertEqual(
            first_result.loc[0, "Likely_Biased_Terms"],
            second_result.loc[0, "Likely_Biased_Terms"],
        )


if __name__ == "__main__":
    unittest.main()
