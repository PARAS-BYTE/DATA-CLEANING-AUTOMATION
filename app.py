"""
AI CSV Cleaner: Streamlit + LangGraph + Groq

Install:
    pip install -U streamlit pandas numpy langchain-core langchain-groq langgraph

Run:
    streamlit run app.py

The Groq API key is entered in the Streamlit sidebar (never stored in code).

Workflow (LangGraph):
    understand -> (clean needed?) -> clean -> execute -> validate -> finish
                                       ^         |          |
                                       +---------+----------+   (retry loop)
"""

import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

# ----------------------------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------------------------
SCRIPT_TIMEOUT_SEC = 120
DEFAULT_MODEL = "openai/gpt-oss-120b"
MODEL_OPTIONS = [DEFAULT_MODEL, "llama-3.3-70b-versatile", "openai/gpt-oss-20b"]

AGENTS = {
    "understand": {
        "icon": "🔍",
        "name": "Understanding Agent",
        "role": "Reads the dataset profile and decides whether cleaning is needed, "
                "listing every issue it finds.",
    },
    "clean": {
        "icon": "🧹",
        "name": "Cleaning Agent",
        "role": "Writes a pandas script that fixes only the issues that were found, "
                "while preserving valid data.",
    },
    "execute": {
        "icon": "⚙️",
        "name": "Execution Engine",
        "role": "Runs the generated script in an isolated process with a timeout "
                "and profiles the output (not an LLM).",
    },
    "validate": {
        "icon": "✅",
        "name": "Validation Agent",
        "role": "Audits the cleaned data against the original, checks for "
                "remaining issues and for accidental data loss.",
    },
}

# ----------------------------------------------------------------------------
# 2. PROMPTS
# ----------------------------------------------------------------------------
UNDERSTANDING_PROMPT = """You are a senior data analyst reviewing a raw CSV dataset profile.

Your job: decide whether the data needs cleaning and list concrete issues.

Check each of these and only report what the profile actually shows:
1. Missing values (per column, and whether the missing rate is meaningful)
2. Duplicate rows
3. Wrong data types (numbers/dates stored as text, IDs stored as numbers, etc.)
4. Dirty text: leading/trailing spaces, inconsistent casing, spelling variants of the
   same category, placeholder strings such as "N/A", "null", "-", "?", "unknown"
5. Numeric or date values stored with symbols or mixed formats (currency, %, commas)
6. Impossible values (negative age, future birth dates, etc.).
   Statistical outliers are NOT errors unless they are clearly impossible.

Rules:
- Base every claim on evidence in the profile. Do not invent issues.
- Never recommend deleting rows or categories without clear evidence they are invalid.
- Do not write code.

Respond in exactly this format:
CLEANING_REQUIRED: YES or NO
SUMMARY: one or two sentences
ISSUES:
- <column or "dataset">: <issue> -> <recommended fix>
(write "- none" if no issues)
"""

CLEANING_PROMPT = """You are a careful pandas data engineer. Write ONE complete Python script.

Predefined variables (already defined above your code, do NOT redefine them):
- INPUT_PATH: path of the original CSV
- OUTPUT_PATH: where the cleaned CSV must be saved
- ENCODING: encoding to use when reading the CSV
Already imported: pandas as pd, numpy as np, re. You may import other standard libraries.

Requirements:
1. Load with: df = pd.read_csv(INPUT_PATH, encoding=ENCODING)
2. Fix ONLY the issues listed in the analysis and the feedback. Do not make extra changes.
3. Preserve valid rows, columns, categories and values.
   Do not drop rare categories or statistical outliers without proof they are invalid.
   Do not rename columns unless the analysis says names are broken.
4. Be safe: use errors="coerce" for numeric/date conversion and then check that you
   did not turn many valid values into NaN. Do not blindly fill missing values; only
   impute when it is clearly justified (e.g. median for a numeric column with few gaps).
5. Add a short comment before each step.
6. After each step, print one line describing what changed (e.g. print("Removed 12 duplicate rows")).
7. Finish with: df.to_csv(OUTPUT_PATH, index=False)
8. If feedback from a previous attempt is provided, fix exactly those problems.

Output ONLY Python code inside a single ```python fenced block. No explanation.
"""

VALIDATION_PROMPT = """You are a strict data-quality auditor.

You receive: the original profile, the cleaned profile, programmatic comparison stats,
the initial analysis, the execution log, and the cleaning history.

Decide:
- Were the issues from the initial analysis actually fixed?
- Is any important issue still present in the cleaned profile?
- Was valid data lost (too many rows dropped, columns removed, categories merged wrongly,
  numeric values turned into NaN)?

Rules:
- Do not require valid statistical outliers to be removed.
- Do not fail for cosmetic reasons. FAIL only for real remaining problems or data loss.
- Use BLOCKED only when continuing automatically would be unsafe.

Respond in exactly this format:
VALIDATION_STATUS: PASS or FAIL or BLOCKED
REASON: one or two sentences
REMAINING_ISSUES:
- <issue and the exact fix needed>
(write "- none" if PASS)
"""


# ----------------------------------------------------------------------------
# 3. DATA HELPERS
# ----------------------------------------------------------------------------
def read_csv_safe(path_or_buffer):
    """Read CSV trying common encodings. Returns (dataframe, encoding)."""
    last_error = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            if hasattr(path_or_buffer, "seek"):
                path_or_buffer.seek(0)
            return pd.read_csv(path_or_buffer, encoding=enc), enc
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise ValueError(f"Could not read the CSV: {last_error}")


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column table shown to the user and used in the profile."""
    rows = []
    for col in df.columns:
        s = df[col]
        examples = s.dropna().astype(str).unique()[:3]
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing %": round(s.isna().mean() * 100, 1),
            "unique": int(s.nunique(dropna=True)),
            "examples": ", ".join(examples),
        })
    return pd.DataFrame(rows)


def dataset_stats(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def profile_dataframe(df: pd.DataFrame) -> str:
    """Compact text profile that the agents read."""
    stats = dataset_stats(df)
    lines = [
        f"Rows: {stats['rows']} | Columns: {stats['columns']} | "
        f"Missing cells: {stats['missing_cells']} | Duplicate rows: {stats['duplicate_rows']}",
        "",
        "COLUMNS (name | dtype | missing | unique | examples)",
    ]
    for _, r in column_summary(df).iterrows():
        lines.append(
            f"- {r['column']} | {r['dtype']} | {r['missing']} ({r['missing %']}%) | "
            f"{r['unique']} | {r['examples']}"
        )

    text_cols = df.select_dtypes(include=["object", "string"]).columns[:25]
    if len(text_cols):
        lines += ["", "TOP VALUES OF TEXT COLUMNS (value: count)"]
        for col in text_cols:
            top = df[col].value_counts(dropna=True).head(6)
            lines.append(f"- {col}: " + "; ".join(f"{k!r}: {v}" for k, v in top.items()))
            blanks = int((df[col].astype(str).str.strip() != df[col].astype(str)).sum())
            if blanks:
                lines.append(f"  (has {blanks} values with leading/trailing spaces)")

    numeric = df.select_dtypes(include=np.number)
    if not numeric.empty:
        lines += ["", "NUMERIC SUMMARY", numeric.describe().round(3).T.to_string()]

    lines += ["", "SAMPLE ROWS", df.head(5).to_string(index=False)]
    return "\n".join(lines)[:18000]


def compare_stats(before: dict, after: dict, cols_before, cols_after) -> str:
    removed = sorted(set(cols_before) - set(cols_after))
    added = sorted(set(cols_after) - set(cols_before))
    return (
        f"Rows: {before['rows']} -> {after['rows']}\n"
        f"Columns: {before['columns']} -> {after['columns']}\n"
        f"Missing cells: {before['missing_cells']} -> {after['missing_cells']}\n"
        f"Duplicate rows: {before['duplicate_rows']} -> {after['duplicate_rows']}\n"
        f"Columns removed: {removed or 'none'}\n"
        f"Columns added: {added or 'none'}"
    )


# ----------------------------------------------------------------------------
# 4. LLM HELPERS
# ----------------------------------------------------------------------------
def make_llm(api_key: str, model: str) -> ChatGroq:
    return ChatGroq(model=model, temperature=0, api_key=api_key)


def ask(llm, system_prompt: str, user_prompt: str) -> str:
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    content = response.content
    if isinstance(content, list):  # some models return content blocks
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return content.strip()


def parse_field(text: str, field: str, allowed: str):
    match = re.search(rf"{field}\s*:\s*\**\s*({allowed})", text, re.I)
    return match.group(1).upper() if match else None


def extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.I | re.S)
    return (match.group(1) if match else text).strip()


# ----------------------------------------------------------------------------
# 5. SCRIPT EXECUTION (subprocess, not exec)
# ----------------------------------------------------------------------------
def run_cleaning_script(code: str, work_dir: Path, input_path: Path, output_path: Path, encoding: str, attempt: int):
    """Run generated code in a separate Python process. Returns (ok, log)."""
    header = (
        "import re\nimport numpy as np\nimport pandas as pd\n"
        f"INPUT_PATH = {str(input_path)!r}\n"
        f"OUTPUT_PATH = {str(output_path)!r}\n"
        f"ENCODING = {encoding!r}\n\n"
    )
    script_path = (work_dir / f"cleaning_script_attempt_{attempt}.py").resolve()
    script_path.write_text(header + code, encoding="utf-8")
    if output_path.exists():
        output_path.unlink()

    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=SCRIPT_TIMEOUT_SEC, cwd=str(work_dir.resolve()),
        )
    except subprocess.TimeoutExpired:
        return False, f"Script timed out after {SCRIPT_TIMEOUT_SEC}s."

    log = (proc.stdout or "").strip()
    if proc.returncode != 0:
        return False, f"{log}\n\nSTDERR:\n{(proc.stderr or '')[-1500:]}".strip()
    if not output_path.exists():
        return False, "Script finished but did not create the output CSV."
    return True, log


# ----------------------------------------------------------------------------
# 6. LANGGRAPH WORKFLOW
# ----------------------------------------------------------------------------
class State(TypedDict, total=False):
    input_path: str
    encoding: str
    max_attempts: int
    original_profile: str
    original_stats: dict
    current_profile: str
    analysis: str
    code: str
    exec_log: str
    comparison: str
    feedback: str
    validation: str
    history: list
    iteration: int
    work_dir: str
    cleaned_df: pd.DataFrame
    status: str
    final_message: str


def build_graph(llm):
    def understand(state: State):
        user = f"Dataset profile:\n\n{state['original_profile']}"
        answer = ask(llm, UNDERSTANDING_PROMPT, user)
        decision = parse_field(answer, "CLEANING_REQUIRED", "YES|NO") or "YES"
        return {"analysis": answer, "status": "CLEAN" if decision == "NO" else "NEEDS_CLEANING"}

    def route_understand(state: State):
        return "finish" if state["status"] == "CLEAN" else "clean"

    def clean(state: State):
        user = (
            f"INITIAL ANALYSIS:\n{state['analysis']}\n\n"
            f"ORIGINAL DATASET PROFILE:\n{state['original_profile']}\n\n"
            f"CLEANING HISTORY:\n{chr(10).join(state.get('history', [])) or 'First attempt.'}\n\n"
            f"FEEDBACK FROM PREVIOUS ATTEMPT (fix these):\n{state.get('feedback') or 'None'}\n\n"
            "Write the full cleaning script now."
        )
        return {"code": extract_code(ask(llm, CLEANING_PROMPT, user))}

    def execute(state: State):
        attempt = state.get("iteration", 0) + 1
        work_dir = Path(state["work_dir"])
        output_path = (work_dir / f"cleaned_attempt_{attempt}.csv").resolve()
        ok, log = run_cleaning_script(
            state["code"], work_dir, Path(state["input_path"]).resolve(), output_path, state["encoding"], attempt
        )
        history = state.get("history", [])

        if ok:
            try:
                cleaned = pd.read_csv(output_path)
                if cleaned.empty:
                    raise ValueError("The cleaned CSV is empty.")
            except Exception as exc:  # noqa: BLE001
                ok, log = False, f"{log}\nOutput problem: {exc}"

        if not ok:
            return {
                "iteration": attempt,
                "exec_log": log,
                "feedback": f"Execution failed:\n{log}",
                "history": history + [f"Attempt {attempt}: FAILED"],
                "status": "RETRY" if attempt < state["max_attempts"] else "BLOCKED",
            }

        original_df, _ = read_csv_safe(state["input_path"])
        comparison = compare_stats(
            state["original_stats"], dataset_stats(cleaned), original_df.columns, cleaned.columns
        )
        return {
            "iteration": attempt,
            "exec_log": log or "(script printed nothing)",
            "cleaned_df": cleaned,
            "current_profile": profile_dataframe(cleaned),
            "comparison": comparison,
            "history": history + [f"Attempt {attempt}: script ran OK, shape={cleaned.shape}"],
            "status": "VALIDATE",
        }

    def route_execute(state: State):
        return {"VALIDATE": "validate", "RETRY": "clean"}.get(state["status"], "finish")

    def validate(state: State):
        user = (
            f"ORIGINAL PROFILE:\n{state['original_profile']}\n\n"
            f"CLEANED PROFILE:\n{state['current_profile']}\n\n"
            f"COMPARISON:\n{state['comparison']}\n\n"
            f"INITIAL ANALYSIS:\n{state['analysis']}\n\n"
            f"EXECUTION LOG:\n{state['exec_log']}\n\n"
            f"HISTORY:\n{chr(10).join(state.get('history', []))}\n\n"
            "Audit the cleaned dataset."
        )
        answer = ask(llm, VALIDATION_PROMPT, user)
        verdict = parse_field(answer, "VALIDATION_STATUS", "PASS|FAIL|BLOCKED") or "FAIL"
        status = {"PASS": "PASS", "BLOCKED": "BLOCKED"}.get(verdict, "RETRY")
        return {"validation": answer, "feedback": answer, "status": status}

    def route_validate(state: State):
        if state["status"] in ("PASS", "BLOCKED"):
            return "finish"
        return "finish" if state["iteration"] >= state["max_attempts"] else "clean"

    def finish(state: State):
        status = state["status"]
        if status == "CLEAN":
            msg = "✅ No cleaning was required. Your data already looks good."
        elif status == "PASS":
            msg = f"✅ Cleaning finished and passed validation in {state['iteration']} attempt(s)."
        elif status == "RETRY":
            msg = "⚠️ Maximum attempts reached. The latest output is shown, but validation still lists issues."
        else:
            msg = "🛑 The workflow was blocked. Review the reports below before using any output."
        return {"final_message": msg}

    g = StateGraph(State)
    g.add_node("understand", understand)
    g.add_node("clean", clean)
    g.add_node("execute", execute)
    g.add_node("validate", validate)
    g.add_node("finish", finish)
    g.add_edge(START, "understand")
    g.add_conditional_edges("understand", route_understand, {"clean": "clean", "finish": "finish"})
    g.add_edge("clean", "execute")
    g.add_conditional_edges("execute", route_execute, {"validate": "validate", "clean": "clean", "finish": "finish"})
    g.add_conditional_edges("validate", route_validate, {"clean": "clean", "finish": "finish"})
    g.add_edge("finish", END)
    return g.compile()


# ----------------------------------------------------------------------------
# 7. UI HELPERS
# ----------------------------------------------------------------------------
def render_agent_event(node: str, update: dict, elapsed: float | None = None):
    """Show what one agent did, as an expandable card."""
    if node == "finish":
        return
    meta = AGENTS[node]
    failed = node == "execute" and update.get("status") in ("RETRY", "BLOCKED")
    attempt = update.get("iteration")
    title = f"{meta['icon']} {meta['name']}"
    if attempt and node in ("execute",):
        title += f" (attempt {attempt})"
    if elapsed is not None:
        title += f" · {elapsed:.1f}s"

    with st.status(title, state="error" if failed else "complete", expanded=False):
        st.caption(meta["role"])
        if node == "understand":
            st.text(update["analysis"])
        elif node == "clean":
            st.code(update["code"], language="python")
        elif node == "execute":
            st.markdown("**Execution log**")
            st.text(update.get("exec_log", ""))
            if update.get("comparison"):
                st.markdown("**Before → after**")
                st.text(update["comparison"])
        elif node == "validate":
            st.text(update["validation"])


def render_agent_roster():
    with st.expander("🤖 Meet the agents"):
        for meta in AGENTS.values():
            st.markdown(f"**{meta['icon']} {meta['name']}**: {meta['role']}")


def render_upload_overview(df: pd.DataFrame):
    """Everything the user can learn about the data right after uploading."""
    stats = dataset_stats(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{stats['rows']:,}")
    c2.metric("Columns", stats["columns"])
    c3.metric("Missing cells", f"{stats['missing_cells']:,}")
    c4.metric("Duplicate rows", f"{stats['duplicate_rows']:,}")

    tab_preview, tab_cols, tab_stats = st.tabs(["Preview", "Column info", "Numeric summary"])
    with tab_preview:
        st.dataframe(df.head(20))
    with tab_cols:
        st.dataframe(column_summary(df), hide_index=True)
    with tab_stats:
        numeric = df.select_dtypes(include=np.number)
        if numeric.empty:
            st.info("No numeric columns found.")
        else:
            st.dataframe(numeric.describe().round(3).T)


def render_results(result: dict, events: list):
    st.subheader("Result")
    status = result.get("status")
    (st.success if status in ("PASS", "CLEAN") else st.warning)(result.get("final_message", "Finished."))

    st.subheader("Agent activity")
    for node, update, elapsed in events:
        render_agent_event(node, update, elapsed)

    cleaned_df = result.get("cleaned_df")
    if cleaned_df is not None:
        st.subheader("Cleaned data")
        before, after = result["original_stats"], dataset_stats(cleaned_df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{after['rows']:,}", after["rows"] - before["rows"])
        c2.metric("Columns", after["columns"], after["columns"] - before["columns"])
        c3.metric("Missing cells", f"{after['missing_cells']:,}", after["missing_cells"] - before["missing_cells"], delta_color="inverse")
        c4.metric("Duplicate rows", f"{after['duplicate_rows']:,}", after["duplicate_rows"] - before["duplicate_rows"], delta_color="inverse")

        tab_data, tab_cols = st.tabs(["Cleaned data", "Column info"])
        with tab_data:
            st.dataframe(cleaned_df)
        with tab_cols:
            st.dataframe(column_summary(cleaned_df), hide_index=True)

        st.download_button(
            "⬇️ Download cleaned CSV",
            data=cleaned_df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_output.csv",
            mime="text/csv",
            type="primary",
        )
    elif status == "CLEAN":
        st.info("The original data is unchanged because no cleaning was needed.")


def run_workflow(graph, initial_state: dict):
    """Stream the graph so each agent's result appears the moment it finishes."""
    final, events = dict(initial_state), []
    live = st.container()
    last = time.time()
    with st.spinner("Agents are working..."):
        for chunk in graph.stream(initial_state, config={"recursion_limit": 60}, stream_mode="updates"):
            for node, update in chunk.items():
                now = time.time()
                final.update(update)
                events.append((node, update, now - last))
                last = now
                with live:
                    render_agent_event(node, update, events[-1][2])
    return final, events


# ----------------------------------------------------------------------------
# 8. MAIN APP
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="AI CSV Cleaner", page_icon="🧹", layout="wide")
    st.title("🧹 AI CSV Cleaner")
    st.write("Upload a CSV. AI agents will inspect, clean and validate it, and you can watch each step.")

    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("Groq API key", type="password", help="Get one at console.groq.com")
        model = st.selectbox("Model", MODEL_OPTIONS)
        max_attempts = st.slider("Max cleaning attempts", 1, 5, 3)
        st.caption("Generated cleaning scripts run on this machine. Use only data you trust.")
        render_agent_roster()

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV file to get started.")
        st.stop()

    try:
        df, encoding = read_csv_safe(uploaded)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    # Forget old results when a different file is uploaded.
    file_key = f"{uploaded.name}-{uploaded.size}"
    if st.session_state.get("file_key") != file_key:
        st.session_state.clear()
        st.session_state["file_key"] = file_key

    st.subheader(f"Your data: {uploaded.name}")
    render_upload_overview(df)

    if st.button("Clean dataset", type="primary", disabled=not api_key):
        profile = profile_dataframe(df)
        st.subheader("Agent activity (live)")
        try:
            graph = build_graph(make_llm(api_key, model))
            # All working files live in a temp folder that is deleted afterwards.
            with tempfile.TemporaryDirectory(prefix="csv_cleaner_") as tmp:
                input_path = Path(tmp) / "input.csv"
                input_path.write_bytes(uploaded.getvalue())
                initial_state = {
                    "input_path": str(input_path),
                    "work_dir": tmp,
                    "encoding": encoding,
                    "max_attempts": max_attempts,
                    "original_profile": profile,
                    "original_stats": dataset_stats(df),
                    "current_profile": profile,
                    "history": [],
                    "iteration": 0,
                    "status": "START",
                }
                result, events = run_workflow(graph, initial_state)
            st.session_state["result"], st.session_state["events"] = result, events
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Workflow failed: {type(exc).__name__}: {exc}")

    if not api_key:
        st.warning("Enter your Groq API key in the sidebar to enable cleaning.")

    if "result" in st.session_state:
        render_results(st.session_state["result"], st.session_state["events"])


if __name__ == "__main__":
    main()