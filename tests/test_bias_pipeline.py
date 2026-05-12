import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd
from openai import AzureOpenAI

from bias_pipeline import (
    _parse_model_response_text,
    AzureBiasPipeline,
    AzureBiasPipelineConfig,
    build_analysis_ready_dataframe,
    build_reviewer_adjudication_dataframe,
    chunk_by_sentences,
    filter_hallucinated_terms,
    postprocess_results,
    term_present_in_chunk,
    write_output_bundle,
)


class BiasPipelineTests(unittest.TestCase):
    @staticmethod
    def _build_fake_client(handler):
        class FakeCompletions:
            def create(self, **kwargs):
                return handler(kwargs)

        return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

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
            return {
                "possible": [{"term": "obese", "categories": ["weight-based identity label"]}],
                "likely": [{"term": "difficult patient", "categories": ["difficult-patient framing"]}],
            }

    def test_chunk_by_sentences_respects_boundaries(self):
        text = "Sentence one. Sentence two is a bit longer. Sentence three."
        chunks = chunk_by_sentences(text, max_chars=30)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(len(chunk) <= 30 for chunk in chunks))
        self.assertEqual(" ".join(chunks), text)

    def test_postprocess_results_filters_physiologic_normal_language(self):
        result = {
            "possible": ["normal", "normal breath sounds", {"term": "obese", "categories": ["weight-based identity label"]}],
            "likely": [{"term": "difficult patient", "categories": ["difficult-patient framing"]}, "mood and affect congruent"],
        }

        filtered = postprocess_results(result)

        self.assertEqual(filtered["possible"], [{"term": "obese", "categories": ["weight-based identity label"]}])
        self.assertEqual(filtered["likely"], [{"term": "difficult patient", "categories": ["difficult-patient framing"]}])

    def test_parse_model_response_text_salvages_first_json_object(self):
        parsed = _parse_model_response_text(
            '{"possible":[{"term":"obese","categories":["weight-based identity label"]}],"likely":[]} trailing text'
        )

        self.assertEqual(
            parsed,
            {"possible": [{"term": "obese", "categories": ["weight-based identity label"]}], "likely": []},
        )

    def test_call_model_on_chunk_prefers_structured_outputs(self):
        captured_kwargs = []

        def handler(kwargs):
            captured_kwargs.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"possible":[],"likely":[{"term":"difficult patient","categories":["difficult-patient framing"]}]}'
                        )
                    )
                ]
            )

        pipeline = AzureBiasPipeline(
            self._build_fake_client(handler),
            AzureBiasPipelineConfig(
                prompt_path="bias_detection_prompt.py",
                model_for_api="test-model",
                cache_chunk_responses=False,
                enable_second_pass_adjudication=False,
            ),
        )

        result = pipeline.call_model_on_chunk("This is a difficult patient.", chunk_index=0)

        self.assertEqual(
            result["likely"],
            [{"term": "difficult patient", "categories": ["difficult-patient framing"]}],
        )
        self.assertEqual(captured_kwargs[0]["response_format"]["type"], "json_schema")
        self.assertTrue(captured_kwargs[0]["response_format"]["json_schema"]["strict"])

    def test_call_model_on_chunk_falls_back_when_structured_outputs_unsupported(self):
        captured_kwargs = []

        def handler(kwargs):
            captured_kwargs.append(kwargs)
            if "response_format" in kwargs:
                raise ValueError("Invalid parameter: response_format")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"possible":[],"likely":[{"term":"difficult patient","categories":["difficult-patient framing"]}]}'
                        )
                    )
                ]
            )

        pipeline = AzureBiasPipeline(
            self._build_fake_client(handler),
            AzureBiasPipelineConfig(
                prompt_path="bias_detection_prompt.py",
                model_for_api="test-model",
                cache_chunk_responses=False,
                enable_second_pass_adjudication=False,
            ),
        )

        first_result = pipeline.call_model_on_chunk("This is a difficult patient.", chunk_index=0)
        second_result = pipeline.call_model_on_chunk("Another difficult patient.", chunk_index=1)

        self.assertEqual(
            first_result["likely"],
            [{"term": "difficult patient", "categories": ["difficult-patient framing"]}],
        )
        self.assertEqual(
            second_result["likely"],
            [{"term": "difficult patient", "categories": ["difficult-patient framing"]}],
        )
        self.assertIn("response_format", captured_kwargs[0])
        self.assertNotIn("response_format", captured_kwargs[1])
        self.assertNotIn("response_format", captured_kwargs[2])

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

    def test_merge_chunk_results_unions_categories_for_same_term(self):
        merged = AzureBiasPipeline._merge_chunk_results(
            ["chunk one", "chunk two"],
            {
                0: {
                    "possible": [],
                    "likely": [{"term": "obese", "categories": ["weight-based identity label"]}],
                },
                1: {
                    "possible": [],
                    "likely": [{"term": "obese", "categories": ["condition identity label", "weight-based identity label"]}],
                },
            },
        )

        self.assertEqual(
            merged["likely"],
            [{"term": "obese", "categories": ["weight-based identity label", "condition identity label"]}],
        )

    def test_process_dataframe_uses_model_categories_when_valid(self):
        class StructuredStub(self.StubPipeline):
            def analyze_note_text(self, full_text: str):
                self.analyze_calls += 1
                return {
                    "possible": [],
                    "likely": [
                        {
                            "term": "was told to",
                            "categories": ["paternalistic framing", "autonomy-undermining language"],
                        }
                    ],
                }

        df = pd.DataFrame([{"unique_id": 1, "note_text": "The patient was told to return in two weeks."}])
        pipeline = StructuredStub()

        processed = pipeline.process_dataframe(df)

        self.assertEqual(
            processed.loc[0, "Likely_Bias_Categories"],
            '["paternalistic framing", "autonomy-undermining language"]',
        )
        self.assertIn('"categories": ["paternalistic framing", "autonomy-undermining language"]', processed.loc[0, "Likely_Bias_Details"])

    def test_notes_with_chunk_failures_are_not_persisted_in_cache(self):
        class FailureStub(self.StubPipeline):
            def analyze_note_text(self, full_text: str):
                self.analyze_calls += 1
                self._last_chunk_failures = [
                    {
                        "note_hash": self._current_note_hash,
                        "chunk_index": 0,
                        "chunk_char_count": len(full_text),
                        "error": "Extra data: line 1 column 10 (char 9)",
                        "response_preview": '{"possible": []} {"likely": []}',
                    }
                ]
                return {"possible": [], "likely": [{"term": "difficult patient", "categories": ["difficult-patient framing"]}]}

        df = pd.DataFrame([{"unique_id": 1, "note_text": "Same note reused"}])

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = f"{temp_dir}/persistent_cache.csv"

            first_pipeline = FailureStub(cache_path=cache_path)
            first_result = first_pipeline.process_dataframe(df)

            second_pipeline = FailureStub(cache_path=cache_path)
            second_result = second_pipeline.process_dataframe(df)

        self.assertEqual(first_pipeline.analyze_calls, 1)
        self.assertEqual(second_pipeline.analyze_calls, 1)
        self.assertEqual(first_result.loc[0, "Chunk_Failure_Count"], 1)
        self.assertEqual(second_result.loc[0, "Chunk_Failure_Count"], 1)


class HallucinationGuardTests(unittest.TestCase):
    def test_present_when_verbatim_match(self):
        self.assertTrue(term_present_in_chunk("obese", "The patient is obese."))

    def test_present_case_insensitive_and_whitespace_collapsed(self):
        self.assertTrue(term_present_in_chunk("Obese", "OBESE"))
        self.assertTrue(term_present_in_chunk("uncontrolled  diabetic", "patient is uncontrolled diabetic"))

    def test_present_when_term_has_parenthetical_tail(self):
        # The LLM sometimes emits "obesity (BMI 30-39.9" — verify the head matches.
        self.assertTrue(term_present_in_chunk("obesity (BMI 30-39.9", "Patient has obesity (E66.9)."))

    def test_absent_when_term_not_in_chunk(self):
        self.assertFalse(term_present_in_chunk("elderly", "Mrs Smith, age 78, presents with hypertension."))
        self.assertFalse(term_present_in_chunk("non-compliant", "Pt takes meds most days."))

    def test_filter_drops_hallucinated_terms_only(self):
        result = {
            "possible": [
                {"term": "obese", "categories": ["weight-based identity label"]},
                {"term": "elderly", "categories": ["ageism"]},
            ],
            "likely": [
                {"term": "non-compliant", "categories": ["disapproval"]},
                {"term": "morbid obesity", "categories": ["weight-based identity label"]},
            ],
        }
        chunk = "Patient is obese and has morbid obesity (BMI 42)."
        filtered, dropped = filter_hallucinated_terms(result, chunk)
        self.assertEqual([d["term"] for d in dropped], ["elderly", "non-compliant"])
        self.assertEqual(filtered["possible"], [{"term": "obese", "categories": ["weight-based identity label"]}])
        self.assertEqual(filtered["likely"], [{"term": "morbid obesity", "categories": ["weight-based identity label"]}])


if __name__ == "__main__":
    unittest.main()
