import io
import json
import os
from typing import List, Tuple

import pandas as pd
import streamlit as st
from openai import AzureOpenAI

from bias_pipeline import AzureBiasPipeline, AzureBiasPipelineConfig


APP_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
APP_CACHE_PATH = os.path.join(APP_OUTPUT_DIR, "streamlit_bias_cache.csv")


def check_password(widget_key_suffix: str = "") -> bool:
    password_key = st.secrets["app_password"]

    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    if "login_attempts" not in st.session_state:
        st.session_state.login_attempts = 0

    if st.session_state.password_correct:
        return True

    password_key_name = f"password_{widget_key_suffix}"

    def password_entered() -> None:
        entered_password = st.session_state[password_key_name]
        if entered_password == password_key:
            st.session_state.password_correct = True
            st.session_state.login_attempts = 0
        else:
            st.session_state.password_correct = False
            st.session_state.login_attempts += 1

    st.text_input(
        "Password",
        type="password",
        on_change=password_entered,
        key=password_key_name,
    )

    if st.session_state.login_attempts > 0:
        st.error(f"Password incorrect. Attempts: {st.session_state.login_attempts}")

    st.write(
        "*Please contact David Liebovitz, MD if you need an updated password for access.*"
    )
    return False


@st.cache_resource
def get_azure_client(endpoint: str, api_key: str, api_version: str) -> AzureOpenAI:
    return AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
    )


def parse_json_list(value) -> List[str]:
    if pd.isna(value) or value in ("", "[]"):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def normalize_term_string(value: str) -> str:
    if pd.isna(value) or not str(value).strip():
        return ""
    return ", ".join(sorted({item.strip().lower() for item in str(value).split(",") if item.strip()}))


def get_note_id_column(df: pd.DataFrame) -> str:
    for candidate in ("note_id", "unique_id", "patient_study_id"):
        if candidate in df.columns:
            return candidate
    return "__row_id__"


def generate_df(
    columns: List[str],
    n_rows: int,
    client: AzureOpenAI,
    model_for_api: str,
) -> Tuple[pd.DataFrame, str]:
    system_prompt = (
        "You are an expert clinical research assistant. Generate a small, fully de-identified "
        "synthetic dataset for testing stigmatizing language analysis in clinical notes. "
        "Return only CSV."
    )
    user_prompt = f"columns: {columns}, number: {n_rows}"

    try:
        response = client.chat.completions.create(
            model=model_for_api,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=2000,
            temperature=0.5,
        )
        generated_content = (response.choices[0].message.content or "").strip()
        data = io.StringIO(generated_content)
        df = pd.read_csv(data, sep=",")
        if not all(column in df.columns for column in columns):
            data.seek(0)
            df = pd.read_csv(data, sep=",", skiprows=1, header=None, names=columns)
        gen_csv = df.to_csv(index=False)
        return df, gen_csv
    except Exception as exc:
        st.warning(f"LLM data generation failed: {exc}")
        return pd.DataFrame(), None


def run_shared_llm_review(
    notes_df: pd.DataFrame,
    client: AzureOpenAI,
    model_for_api: str,
    api_version: str,
) -> pd.DataFrame:
    os.makedirs(APP_OUTPUT_DIR, exist_ok=True)
    pipeline = AzureBiasPipeline(
        client,
        AzureBiasPipelineConfig(
            prompt_path=os.path.join(os.path.dirname(__file__), "bias_detection_prompt.py"),
            model_for_api=model_for_api,
            temperature=1 if "gpt-5" in model_for_api.lower() else 0,
            cache_path=APP_CACHE_PATH,
            enable_second_pass_adjudication=True,
        ),
    )
    reviewed = pipeline.process_dataframe(notes_df)
    reviewed["llm_stigmatizing_terms"] = reviewed.apply(
        lambda row: ", ".join(
            list(
                dict.fromkeys(
                    parse_json_list(row.get("Likely_Biased_Terms"))
                    + parse_json_list(row.get("Possible_Biased_Terms"))
                )
            )
        ),
        axis=1,
    )
    reviewed["llm_likely_terms_display"] = reviewed["Likely_Biased_Terms"].apply(
        lambda value: ", ".join(parse_json_list(value))
    )
    reviewed["llm_possible_terms_display"] = reviewed["Possible_Biased_Terms"].apply(
        lambda value: ", ".join(parse_json_list(value))
    )
    reviewed["llm_categories_display"] = reviewed.apply(
        lambda row: ", ".join(
            list(
                dict.fromkeys(
                    parse_json_list(row.get("Likely_Bias_Categories"))
                    + parse_json_list(row.get("Possible_Bias_Categories"))
                )
            )
        ),
        axis=1,
    )
    reviewed["llm_prompt_version"] = reviewed.get("Prompt_Version", api_version)
    return reviewed


st.title("Landsburg Society Research Award - AI and Stigmatizing Language")
st.subheader("Drs. Asantewaa Ture & David Liebovitz, Northwestern University")

if check_password():
    st.markdown(
        """
        **Background:**
        Medical records often contain stigmatizing language (e.g., "non-compliant", "refused"), which propagates bias and can be perpetuated/transmitted by AI/LLM models such as Nuance DAX.
        This app enables comparison of physician vs AI-generated notes and the efficacy of manual vs LLM-based stigma term detection, supporting generation of a sharable stigmatizing terms list for downstream AI safeguarding.

        **Security:**
        Only de-identified note text may be processed. For demonstration, dummy notes can be generated below. All LLM review uses the shared Azure pipeline used by the batch runner.
        """
    )

    st.sidebar.header("Compliance & Configuration")
    st.sidebar.markdown(
        """
        - **All data must be fully de-identified.**
        - AI analysis is performed via enterprise Azure OpenAI.
        - The app uses the shared bias pipeline with caching, categorization, and second-pass adjudication.
        """
    )

    openai_api_key = st.secrets["azure_openai_api_key"]
    openai_base_url = st.secrets["azure_openai_endpoint"]
    api_version = st.secrets["api_version"]
    model_for_api = st.secrets["azure_deployment"]
    client = get_azure_client(openai_base_url, openai_api_key, api_version)

    st.sidebar.write(f"Deployment: `{model_for_api}`")

    tabs = st.tabs(
        [
            "Project Overview",
            "Upload or Generate Notes",
            "Manual Review",
            "LLM Review",
            "Compare Results",
            "Export Stigmatizing Terms",
        ]
    )

    with tabs[0]:
        st.markdown(
            """
            ### Project Aims
            1. **Compare** AI- vs physician-generated notes for stigmatizing language.
            2. **Compare** manual versus Azure LLM-based detection of stigmatizing terms.
            3. **Generate** exportable term lists for Nuance DAX and other stakeholders.

            ### Workflow
            - **Step 1:** Upload de-identified data or generate dummy notes.
            - **Step 2:** Review notes manually for stigmatizing content.
            - **Step 3:** Analyze notes via the shared Azure bias pipeline.
            - **Step 4:** Compare results and export insights.
            """
        )

    with tabs[1]:
        st.markdown("#### 1. Upload Deidentified Notes or Generate Sample Data")
        uploaded_file = st.file_uploader(
            "Upload CSV (must have: note_id, author_type, note_text)", type="csv"
        )
        columns = ["note_id", "author_type", "note_text"]
        notes_df = None
        if uploaded_file:
            notes_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded {notes_df.shape[0]} notes.")
            st.dataframe(notes_df.head())
        else:
            st.markdown("**Or generate synthetic clinical notes:**")
            n_rows = st.slider(
                "Number of dummy notes to generate",
                min_value=5,
                max_value=100,
                value=10,
            )
            if st.button("Generate test notes"):
                notes_df, gen_csv = generate_df(columns, n_rows, client, model_for_api)
                if not notes_df.empty:
                    st.success(f"Generated {notes_df.shape[0]} dummy notes.")
                    st.dataframe(notes_df.head())
                    st.download_button(
                        "Download dummy notes as CSV", gen_csv, "dummy_notes.csv"
                    )

        if notes_df is not None:
            notes_df = notes_df.copy()
            if "__row_id__" not in notes_df.columns:
                notes_df["__row_id__"] = range(len(notes_df))
            st.session_state["notes_df"] = notes_df

    with tabs[2]:
        st.markdown("#### 2. Manual Review for Stigmatizing Language")
        st.info(
            "Select individual notes and manually flag stigmatizing terms. Use published definitions and reviewer guidance."
        )
        notes_df = st.session_state.get("notes_df")
        if notes_df is not None:
            note_id_col = get_note_id_column(notes_df)
            idx = st.selectbox("Select a note index for manual review:", notes_df.index)
            selected_note = notes_df.iloc[idx]
            st.write(f"**Note ID**: {selected_note[note_id_col]}")
            if "author_type" in selected_note:
                st.write(f"**Author Type**: {selected_note['author_type']}")
            st.text_area("Note Text", selected_note["note_text"], disabled=True)
            manual_terms = st.text_input(
                "Stigmatizing terms found (comma separated):",
                value=selected_note.get("manual_stigmatizing_terms", ""),
                key=f"manual_terms_{idx}",
            )
            if manual_terms:
                notes_df.loc[idx, "manual_stigmatizing_terms"] = manual_terms
                st.session_state["notes_df"] = notes_df
                st.success("Terms saved for this note.")

    with tabs[3]:
        st.markdown("#### 3. LLM Review via Shared Azure Pipeline")
        notes_df = st.session_state.get("notes_df")
        if notes_df is not None and st.button("Analyze notes with shared pipeline"):
            reviewed = run_shared_llm_review(notes_df, client, model_for_api, api_version)
            st.session_state["notes_df"] = reviewed
            st.success("LLM review complete.")
            preview_cols = [
                column
                for column in (
                    get_note_id_column(reviewed),
                    "note_text",
                    "llm_likely_terms_display",
                    "llm_possible_terms_display",
                    "llm_categories_display",
                )
                if column in reviewed.columns
            ]
            st.dataframe(reviewed[preview_cols].head())

    with tabs[4]:
        st.markdown("#### 4. Compare Manual vs LLM Results")
        notes_df = st.session_state.get("notes_df")
        if (
            notes_df is not None
            and "manual_stigmatizing_terms" in notes_df.columns
            and "llm_stigmatizing_terms" in notes_df.columns
        ):
            note_id_col = get_note_id_column(notes_df)
            comp_df = notes_df[
                [note_id_col, "manual_stigmatizing_terms", "llm_stigmatizing_terms"]
            ].fillna("")
            st.dataframe(comp_df.head(10))
            total = comp_df.shape[0]
            manual_hits = comp_df["manual_stigmatizing_terms"].astype(bool).sum()
            llm_hits = comp_df["llm_stigmatizing_terms"].astype(bool).sum()
            exact_matches = (
                comp_df["manual_stigmatizing_terms"].map(normalize_term_string)
                == comp_df["llm_stigmatizing_terms"].map(normalize_term_string)
            ).sum()
            st.write("**Concordance Metrics:**")
            st.write(f"Notes flagged by manual review: {manual_hits} / {total}")
            st.write(f"Notes flagged by LLM: {llm_hits} / {total}")
            st.write(f"Exact set matches (manual vs LLM): {exact_matches} / {total}")

            if "Likely_Bias_Count" in notes_df.columns and "Possible_Bias_Count" in notes_df.columns:
                st.write(
                    f"Total likely bias flags: {int(notes_df['Likely_Bias_Count'].sum())}"
                )
                st.write(
                    f"Total possible bias flags: {int(notes_df['Possible_Bias_Count'].sum())}"
                )

    with tabs[5]:
        st.markdown("#### 5. Export Sharable List of Stigmatizing Terms")
        notes_df = st.session_state.get("notes_df")
        if notes_df is not None and (
            "manual_stigmatizing_terms" in notes_df.columns
            or "Likely_Biased_Terms_Normalized" in notes_df.columns
            or "Possible_Biased_Terms_Normalized" in notes_df.columns
        ):
            all_terms = []
            if "manual_stigmatizing_terms" in notes_df.columns:
                for terms in notes_df["manual_stigmatizing_terms"].dropna():
                    all_terms.extend(
                        [term.strip() for term in str(terms).split(",") if term.strip()]
                    )
            for column in (
                "Likely_Biased_Terms_Normalized",
                "Possible_Biased_Terms_Normalized",
            ):
                if column in notes_df.columns:
                    for value in notes_df[column].dropna():
                        all_terms.extend(parse_json_list(value))

            unique_terms = sorted({term for term in all_terms if term})
            st.write("**Unique stigmatizing terms found in dataset:**")
            st.write(unique_terms)
            terms_txt = "\n".join(unique_terms)
            st.download_button(
                "Download terms as .txt", terms_txt, "stigmatizing_terms.txt"
            )

    st.markdown(
        """
        ---
        #### References
        1. Weiner SG et al., J Addict Med. 2023;17(4):424-430.
        2. Tate S. J Addict Med. 2024 Jan-Feb 01;18(1):90.
        3. Omiye JA et al. NPJ Digit Med. 2023;6(1):195.

        **For research/educational use only. IRB review required for real data.**
        """
    )
else:
    st.markdown(
        """
        **Background:**
        Medical records often contain stigmatizing language (e.g., "non-compliant", "refused"), which propagates bias and can be perpetuated/transmitted by AI/LLM models such as Nuance DAX.

        Please enter the password above to access the application.
        """
    )
