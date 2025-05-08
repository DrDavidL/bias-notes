import streamlit as st
import pandas as pd
import io
from typing import List, Tuple

# Import Azure GPT connector
from langchain_openai import AzureChatOpenAI

def check_password(widget_key_suffix="") -> bool:
    """
    Check if the entered password is correct and manage login state.
    Also resets the app when a user successfully logs in.
    
    Args:
        widget_key_suffix: A suffix to add to the widget key to avoid duplicate widget IDs
    """
    # Initialize session state variables
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    if "login_attempts" not in st.session_state:
        st.session_state.login_attempts = 0

    # If already authenticated, return True
    if st.session_state.password_correct:
        return True

    # Create a unique key for this password input
    password_key_name = f"password_{widget_key_suffix}"
    
    def password_entered() -> None:
        """Callback function when password is entered."""
        entered_password = st.session_state[password_key_name]
        if entered_password == st.secrets["app_password"]:
            st.session_state.password_correct = True
            st.session_state.login_attempts = 0
        else:
            st.session_state.password_correct = False
            st.session_state.login_attempts += 1

    # Check if password is correct
    if not st.session_state.password_correct:
        st.text_input(
            "Password", type="password", on_change=password_entered, key=password_key_name
        )

        if st.session_state.login_attempts > 0:
            st.error(
                f"😕 Password incorrect. Attempts: {st.session_state.login_attempts}"
            )

        st.write(
            "*Please contact David Liebovitz, MD if you need an updated password for access.*"
        )
        return False

    return True

# --- HEADER ---
if check_password():
    st.title("Landsburg Society Research Award - AI and Stigmatizing Language")
st.subheader("Drs. Asantewaa Ture & David Liebovitz, Northwestern University")
st.markdown("""
**Background:**
Medical records often contain stigmatizing language (e.g., "non-compliant", "refused"), which propagates bias and can be perpetuated/transmitted by AI/LLM models such as Nuance DAX.
This app enables comparison of physician vs AI-generated notes and the efficacy of manual vs LLM-based stigma term detection, supporting generation of a sharable stigmatizing terms list for downstream AI safeguarding.

**Security:**  
Only de-identified note text may be processed. For demonstration, dummy notes can be generated (see below). All LLM queries use enterprise Azure OpenAI endpoints with no data retention per IRB and Northwestern requirements.
""")

# --- SIDEBAR: Compliance Notices & Config ---
st.sidebar.header("Compliance & Configuration")
st.sidebar.markdown("""
- **All data must be fully de-identified.**
- AI analysis performed via secure Azure OpenAI enterprise connection (no data retention).
- [Read more: Northwestern GenAI Guidance](https://www.feinberg.northwestern.edu/compliance/genai.html)
""")
selected_model = st.sidebar.selectbox("Select Azure OpenAI Model", ["gpt-4o", "gpt-3.5-turbo"], index=0)

# Azure OpenAI secrets (get these from Streamlit secrets manager)
openai_api_key = st.secrets["azure_openai_api_key"]
openai_base_url = st.secrets["azure_openai_endpoint"]

# --- DUMMY DATA GENERATOR (LLM-assisted, customizable columns/rows) ---
def generate_df(columns: List[str], n_rows: int, selected_model: str) -> Tuple[pd.DataFrame, str]:
    """
    Generates a synthetic dataframe based on column names and number of rows using an LLM.
    """
    # You must define your dataframe_generation_system_prompt in a 'prompts' module or here:
    system_prompt = "You are an expert clinical research assistant. Please create a small, realistically de-identified dataset for testing stigmatizing language analysis in clinical notes. Format as CSV. Example columns: 'note_id', 'author_type', 'note_text'."
    user_prompt = f"columns: {columns}, number: {n_rows}"
    llm = AzureChatOpenAI(
        azure_deployment=st.secrets["azure_deployment"],
        api_version=st.secrets["api_version"],
        azure_endpoint=openai_base_url,
        api_key=openai_api_key,
        temperature=0.5,
        max_tokens=4000,
        timeout=None,
        max_retries=2,
        model_kwargs={"seed": 42},
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        response = llm.invoke(messages)
        generated_content = response.content if hasattr(response, "content") else str(response)
        data = io.StringIO(generated_content)
        # Try reading with header
        df = pd.read_csv(data, sep=",")
        if not all(col in df.columns for col in columns):
            data.seek(0)
            df = pd.read_csv(data, sep=",", skiprows=1, header=None, names=columns)
        gen_csv = df.to_csv(index=False)
        return df, gen_csv
    except Exception as e:
        st.warning(f"LLM data generation failed: {e}")
        return pd.DataFrame(), None

# --- MAIN TAB LAYOUT ---
tabs = st.tabs([
    "Project Overview",
    "Upload or Generate Notes",
    "Manual Review",
    "LLM Review",
    "Compare Results",
    "Export Stigmatizing Terms"
])

# ---- 1. Project Overview ----
with tabs[0]:
    st.markdown("""
### Project Aims
1. **Compare** AI- vs physician-generated notes for stigmatizing language.
2. **Compare** manual versus Azure LLM-based detection of stigmatizing terms.
3. **Generate** exportable term lists for Nuance DAX and other stakeholders.
    
### Workflow
- **Step 1:** Upload de-identified data *or* generate dummy notes.
- **Step 2:** Review notes manually for stigmatizing content.
- **Step 3:** Analyze notes via LLM (Azure GPT).
- **Step 4:** Compare results; export insights and term lists.
    """)

# ---- 2. Upload or Generate Notes ----
with tabs[1]:
    st.markdown("#### 1. Upload Deidentified Notes or Generate Sample Data")
    uploaded_file = st.file_uploader("Upload CSV (must have: note_id, author_type, note_text)", type="csv")
    columns = ["note_id", "author_type", "note_text"]
    notes_df = None
    if uploaded_file:
        notes_df = pd.read_csv(uploaded_file)
        st.success(f"Uploaded {notes_df.shape[0]} notes.")
        st.dataframe(notes_df.head())
    else:
        st.markdown("**Or generate synthetic clinical notes:**")
        n_rows = st.slider("Number of dummy notes to generate", min_value=5, max_value=100, value=10)
        if st.button("Generate test notes"):
            notes_df, gen_csv = generate_df(columns, n_rows, selected_model)
            if not notes_df.empty:
                st.success(f"Generated {notes_df.shape[0]} dummy notes.")
                st.dataframe(notes_df.head())
                st.download_button("Download dummy notes as CSV", gen_csv, "dummy_notes.csv")

    # Store for use in session
    if notes_df is not None:
        st.session_state["notes_df"] = notes_df

# ---- 3. Manual Review ----
with tabs[2]:
    st.markdown("#### 2. Manual Review for Stigmatizing Language")
    st.info("Select individual notes and manually flag stigmatizing terms. Use published definitions and training (e.g., ADA, NIDA).")
    notes_df = st.session_state.get("notes_df")
    if notes_df is not None:
        idx = st.selectbox("Select a note index for manual review:", notes_df.index)
        selected_note = notes_df.iloc[idx]
        st.write(f"**Note ID**: {selected_note['note_id']}")
        st.write(f"**Author Type**: {selected_note['author_type']}")
        st.text_area("Note Text", selected_note['note_text'], disabled=True)
        manual_terms = st.text_input("Stigmatizing terms found (comma separated):", key=f"manual_terms_{idx}")
        if manual_terms:
            notes_df.loc[idx, "manual_stigmatizing_terms"] = manual_terms
            st.success("Terms saved for this note.")

# ---- 4. LLM Review ----
with tabs[3]:
    st.markdown("#### 3. LLM (Azure OpenAI) Review for Stigmatizing Language")
    notes_df = st.session_state.get("notes_df")
    if notes_df is not None and st.button("Analyze notes with LLM"):
        # Run through each note
        result_terms = []
        for idx, row in notes_df.iterrows():
            prompt = (f"Review the following de-identified clinical progress note. "
                      "Identify all stigmatizing terms or phrases, as defined by published lists (e.g., ADA, NIDA):\n"
                      f"{row['note_text']}\n\n"
                      "Return only a comma-separated list of stigmatizing terms (leave blank if none).")
            llm = AzureChatOpenAI(
                azure_deployment=st.secrets["azure_deployment"],
                api_version=st.secrets["api_version"],
                azure_endpoint=openai_base_url,
                api_key=openai_api_key,
                temperature=0.2,
                max_tokens=256,
                timeout=60,
                max_retries=2,
            )
            response = llm.invoke([
                {"role": "system", "content": "You are an expert in bias and language in medical notes."},
                {"role": "user", "content": prompt}
            ])
            content = response.content.strip() if hasattr(response, "content") else str(response).strip()
            result_terms.append(content)
        notes_df["llm_stigmatizing_terms"] = result_terms
        st.session_state["notes_df"] = notes_df
        st.success("LLM review complete.")
        st.dataframe(notes_df[["note_id", "note_text", "llm_stigmatizing_terms"]].head())

# ---- 5. Compare Results ----
with tabs[4]:
    st.markdown("#### 4. Compare Manual vs LLM Results")
    notes_df = st.session_state.get("notes_df")
    if notes_df is not None and "manual_stigmatizing_terms" in notes_df.columns and "llm_stigmatizing_terms" in notes_df.columns:
        comp_df = notes_df[["note_id", "manual_stigmatizing_terms", "llm_stigmatizing_terms"]].fillna("")
        st.dataframe(comp_df.head(10))
        st.write("**Concordance Metrics:**")
        total = comp_df.shape[0]
        manual_hits = comp_df["manual_stigmatizing_terms"].astype(bool).sum()
        llm_hits = comp_df["llm_stigmatizing_terms"].astype(bool).sum()
        both_hits = (comp_df["manual_stigmatizing_terms"] == comp_df["llm_stigmatizing_terms"]).sum()
        st.write(f"Notes flagged by manual review: {manual_hits} / {total}")
        st.write(f"Notes flagged by LLM: {llm_hits} / {total}")
        st.write(f"Exact matches (manual vs LLM): {both_hits} / {total}")

# ---- 6. Export Stigmatizing Terms ----
with tabs[5]:
    st.markdown("#### 5. Export Sharable List of Stigmatizing Terms")
    notes_df = st.session_state.get("notes_df")
    if notes_df is not None and ("manual_stigmatizing_terms" in notes_df.columns or "llm_stigmatizing_terms" in notes_df.columns):
        # Aggregate unique terms found (manual and LLM)
        all_terms = []
        for col in ["manual_stigmatizing_terms", "llm_stigmatizing_terms"]:
            if col in notes_df.columns:
                for terms in notes_df[col].dropna():
                    all_terms += [t.strip() for t in terms.split(",") if t.strip()]
        unique_terms = sorted(set(all_terms))
        st.write("**Unique stigmatizing terms found in dataset:**")
        st.write(unique_terms)
        terms_txt = "\n".join(unique_terms)
        st.download_button("Download terms as .txt", terms_txt, "stigmatizing_terms.txt")

# --- References & Acknowledgements (Footer) ---
st.markdown("""
---
#### References
1. Weiner SG et al., J Addict Med. 2023;17(4):424-430.
2. Tate S. J Addict Med. 2024 Jan-Feb 01;18(1):90.
3. Omiye JA et al. NPJ Digit Med. 2023;6(1):195.

**For research/educational use only. IRB review required for real data.**
""")
