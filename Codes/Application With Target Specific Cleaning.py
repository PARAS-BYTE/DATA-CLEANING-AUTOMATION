import os
import sys
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# Import the core multi-agent engine
from cleaning_engine import (
    stream_data_cleaning,
    DEFAULT_GROQ_API_KEY,
    DEFAULT_MODEL,
    summarize_csv,
)
from sample_data import generate_sample_dirty_dataset

# ============================================================
# PAGE CONFIGURATION & METADATA
# ============================================================
st.set_page_config(
    page_title="AutoClean AI | Multi-Agent Data Cleaning",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM STYLING & ANIMATIONS
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(20, 24, 40, 0.95) 0%, rgba(30, 15, 45, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 24px;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(12px);
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        color: #0b111e;
        font-weight: 700;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        padding: 4px 12px;
        border-radius: 20px;
        margin-bottom: 12px;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #ffffff 30%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        max-width: 800px;
        line-height: 1.5;
        margin: 0;
    }

    /* Animated Stepper Pipeline */
    .pipeline-wrapper {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 24px;
    }
    .pipeline-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #64748b;
        font-weight: 700;
        margin-bottom: 14px;
    }
    .pipeline-steps {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .step-card {
        flex: 1;
        min-width: 140px;
        padding: 12px 14px;
        border-radius: 10px;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    .step-card.active {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.25) 0%, rgba(147, 51, 234, 0.25) 100%);
        border: 1.5px solid #818cf8;
        box-shadow: 0 0 20px rgba(129, 140, 248, 0.4);
        transform: translateY(-2px);
        animation: pulseBorder 2s infinite;
    }
    .step-card.done {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid #10b981;
    }
    .step-card.failed {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
    }
    .step-icon {
        font-size: 1.3rem;
        margin-bottom: 4px;
        display: block;
    }
    .step-name {
        font-size: 0.8rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    .step-status {
        font-size: 0.7rem;
        color: #94a3b8;
        margin-top: 2px;
    }

    @keyframes pulseBorder {
        0% { box-shadow: 0 0 10px rgba(129, 140, 248, 0.3); }
        50% { box-shadow: 0 0 24px rgba(129, 140, 248, 0.7); }
        100% { box-shadow: 0 0 10px rgba(129, 140, 248, 0.3); }
    }

    /* Agent Output Card */
    .agent-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        transition: transform 0.2s ease;
    }
    .agent-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        border-bottom: 1px solid #1f2937;
        padding-bottom: 10px;
    }
    .agent-title {
        font-weight: 700;
        font-size: 1.05rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .agent-badge {
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 12px;
        letter-spacing: 0.5px;
    }
    .badge-pass { background: #064e3b; color: #34d399; border: 1px solid #059669; }
    .badge-fail { background: #7f1d1d; color: #f87171; border: 1px solid #dc2626; }
    .badge-blocked { background: #78350f; color: #fbbf24; border: 1px solid #d97706; }
    .badge-working { background: #1e1b4b; color: #a5b4fc; border: 1px solid #6366f1; }

    /* Metric Cards */
    .metric-box {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f8fafc;
    }
    .metric-lbl {
        font-size: 0.78rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    .metric-delta {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .delta-positive { color: #10b981; }
    .delta-negative { color: #ef4444; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# INITIALIZE SESSION STATE
# ============================================================
if "cleaned_output_path" not in st.session_state:
    st.session_state.cleaned_output_path = None
if "current_dataset_path" not in st.session_state:
    st.session_state.current_dataset_path = None
if "execution_log" not in st.session_state:
    st.session_state.execution_log = []
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "active_node" not in st.session_state:
    st.session_state.active_node = None
if "completed_nodes" not in st.session_state:
    st.session_state.completed_nodes = set()
if "node_outputs" not in st.session_state:
    st.session_state.node_outputs = {}
if "workflow_finished" not in st.session_state:
    st.session_state.workflow_finished = False

# ============================================================
# SIDEBAR CONFIGURATION
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Autonomous Multi-Agent Config")

    initial_key = ""
    if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
        initial_key = st.secrets["GROQ_API_KEY"]
    elif "GROQ_API_KEY" in os.environ:
        initial_key = os.environ["GROQ_API_KEY"]
    else:
        initial_key = DEFAULT_GROQ_API_KEY

    api_key_input = st.text_input(
        "Groq API Key",
        value=initial_key,
        type="password",
        help="Groq API key used by ChatGroq to power the agents.",
    )

    model_choice = st.selectbox(
        "LLM Model",
        [
            "openai/gpt-oss-120b",
            "meta-llama/llama-prompt-guard-2-22m",
            "meta-llama/llama-prompt-guard-2-86m",
            "qwen/qwen3.6-27b",
            "qwen/qwen3.8-27b",

        ],
        index=0,
        help="Select the model to run Understanding, Cleaning, and Validation agents.",
    )

    recursion_limit = st.slider(
        "Max Iterations Safety Cap",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="LangGraph recursion limit to prevent infinite loops.",
    )

    st.markdown("---")
    st.markdown("### 📋 Multi-Agent Architecture")
    st.caption(
        """
        1. **Dataset Profiler**: Deep statistical & structural scan
        2. **Understanding Agent**: Analyzes flaws & decides cleaning need
        3. **Cleaning Agent**: Crafts executable Pandas script
        4. **Code Execution Engine**: Safely runs script locally
        5. **Validation Auditor**: Checks data integrity & categorical drift
        6. **Autonomous Loop**: Auto-corrects if validation fails!
        """
    )

    if st.button("🔄 Reset App State", use_container_width=True):
        st.session_state.cleaned_output_path = None
        st.session_state.execution_log = []
        st.session_state.active_node = None
        st.session_state.completed_nodes = set()
        st.session_state.node_outputs = {}
        st.session_state.workflow_finished = False
        st.session_state.is_running = False
        st.rerun()

# ============================================================
# HERO BANNER
# ============================================================
st.markdown(
    """
    <div class="hero-container">
        <span class="hero-badge">AI Autonomous Multi-Agent System</span>
        <h1 class="hero-title">Intelligent Data Cleaning Pipeline</h1>
        <p class="hero-subtitle">
            Experience real-time multi-agent orchestration. Observe LLM agents reason, generate Python code,
            execute transformations, and rigorously audit data integrity with automatic feedback loops.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATASET INGESTION SECTION
# ============================================================
st.subheader("📂 1. Select or Upload Dataset")

col_upload, col_sample = st.columns([3, 2])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload any CSV dataset to clean",
        type=["csv"],
        help="Upload your raw or messy CSV file to start the multi-agent cleaning process.",
    )

with col_sample:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    if st.button("⚡ Generate & Load Demo Messy Dataset", use_container_width=True):
        sample_path = generate_sample_dirty_dataset("sample_dirty_data.csv")
        st.session_state.current_dataset_path = sample_path
        st.session_state.cleaned_output_path = None
        st.session_state.execution_log = []
        st.session_state.active_node = None
        st.session_state.completed_nodes = set()
        st.session_state.node_outputs = {}
        st.session_state.workflow_finished = False
        st.success("Loaded demo messy customer dataset with missing values, messy formats, and duplicates!")

# Handle uploaded file
if uploaded_file is not None:
    temp_path = Path("uploaded_raw_data.csv")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.current_dataset_path = str(temp_path.resolve())

# Verify dataset availability
raw_csv_path = st.session_state.get("current_dataset_path")

if not raw_csv_path or not Path(raw_csv_path).exists():
    st.info("👆 Please upload a CSV file or click 'Generate & Load Demo Messy Dataset' above to proceed.")
    st.stop()

# Load raw dataset for exploration
raw_df = pd.read_csv(raw_csv_path)

# ============================================================
# BEFORE CSV INSPECTION
# ============================================================
st.markdown("---")
st.subheader("📊 Raw Dataset Overview (Before Cleaning)")

c1, c2, c3, c4 = st.columns(4)
raw_rows, raw_cols = raw_df.shape
raw_missing = int(raw_df.isna().sum().sum())
raw_duplicates = int(raw_df.duplicated().sum())

with c1:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-val">{raw_rows:,}</div>
            <div class="metric-lbl">Total Rows</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-val">{raw_cols}</div>
            <div class="metric-lbl">Total Columns</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-val" style="color: {'#ef4444' if raw_missing > 0 else '#10b981'};">{raw_missing:,}</div>
            <div class="metric-lbl">Missing Values</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-val" style="color: {'#f59e0b' if raw_duplicates > 0 else '#10b981'};">{raw_duplicates:,}</div>
            <div class="metric-lbl">Duplicate Rows</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("🔍 Preview Raw Dataset & Column Data Types", expanded=True):
    st.dataframe(raw_df, use_container_width=True, height=220)
    col_info = pd.DataFrame(
        {
            "Data Type": raw_df.dtypes.astype(str),
            "Missing Count": raw_df.isna().sum(),
            "Missing %": (raw_df.isna().sum() / len(raw_df) * 100).round(2),
            "Unique Values": raw_df.nunique(),
        }
    )
    st.caption("Column Integrity Breakdown:")
    st.dataframe(col_info.T, use_container_width=True)

# ============================================================
# PIPELINE STEPPER FUNCTION
# ============================================================
pipeline_steps_def = [
    {"id": "profile_original", "name": "Dataset Profiler", "icon": "📊"},
    {"id": "understanding_agent", "name": "Understanding Agent", "icon": "🧠"},
    {"id": "cleaning_agent", "name": "Cleaning Agent", "icon": "💻"},
    {"id": "execute_cleaning", "name": "Code Execution", "icon": "⚙️"},
    {"id": "validation_agent", "name": "Validation Auditor", "icon": "🛡️"},
    {"id": "finalize", "name": "Finalizer", "icon": "🏁"},
]

def render_pipeline_stepper(active_node: str, completed_nodes: set):
    steps_html = []
    for step in pipeline_steps_def:
        s_id = step["id"]
        is_active = (s_id == active_node)
        is_done = (s_id in completed_nodes)

        card_class = "step-card"
        status_text = "Pending"
        if is_active:
            card_class += " active"
            status_text = "⚡ In Progress..."
        elif is_done:
            card_class += " done"
            status_text = "✓ Completed"

        steps_html.append(
            f"""
            <div class="{card_class}">
                <span class="step-icon">{step['icon']}</span>
                <div class="step-name">{step['name']}</div>
                <div class="step-status">{status_text}</div>
            </div>
            """
        )

    full_html = f"""
    <div class="pipeline-wrapper">
        <div class="pipeline-title">🚀 Autonomous Multi-Agent Execution Pipeline</div>
        <div class="pipeline-steps">
            {''.join(steps_html)}
        </div>
    </div>
    """
    return full_html

# Placeholder for the animated stepper
stepper_placeholder = st.empty()
stepper_placeholder.markdown(
    render_pipeline_stepper(st.session_state.active_node, st.session_state.completed_nodes),
    unsafe_allow_html=True,
)

# ============================================================
# RUN WORKFLOW TRIGGER
# ============================================================
st.markdown("---")
col_btn, col_info_btn = st.columns([2, 3])

with col_btn:
    start_cleaning = st.button(
        "🚀 Start Autonomous Multi-Agent Cleaning",
        type="primary",
        disabled=st.session_state.is_running,
        use_container_width=True,
    )

with col_info_btn:
    if st.session_state.workflow_finished:
        st.success("🎉 Workflow successfully completed! Review the step-by-step audit and cleaned dataset below.")
    else:
        st.info("💡 Clicking the button initiates the autonomous LangGraph loop with real-time agent visualization.")

# Containers for live outputs
live_status_placeholder = st.empty()
agent_logs_container = st.container()

# ============================================================
# STREAMING EXECUTION HANDLER
# ============================================================
if start_cleaning:
    st.session_state.is_running = True
    st.session_state.workflow_finished = False
    st.session_state.completed_nodes = set()
    st.session_state.node_outputs = {}
    st.session_state.execution_log = []
    output_clean_path = "cleaned_output.csv"

    # Animated progress bar
    progress_bar = st.progress(0, text="Initializing LangGraph Multi-Agent Workflow...")

    step_progress_weights = {
        "profile_original": 15,
        "understanding_agent": 35,
        "cleaning_agent": 55,
        "execute_cleaning": 75,
        "profile_cleaned": 85,
        "validation_agent": 95,
        "finalize": 100,
    }

    try:
        stream = stream_data_cleaning(
            csv_path=raw_csv_path,
            api_key=api_key_input.strip(),
            model_name=model_choice,
            output_filename=output_clean_path,
            recursion_limit=recursion_limit,
        )

        for event in stream:
            node = event["node"]
            output = event["output"]
            st.session_state.active_node = node
            st.session_state.completed_nodes.add(node)
            st.session_state.node_outputs[node] = output
            st.session_state.execution_log.append(event)

            # Update Stepper with active state animation
            stepper_placeholder.markdown(
                render_pipeline_stepper(node, st.session_state.completed_nodes),
                unsafe_allow_html=True,
            )

            # Update progress bar
            pct = step_progress_weights.get(node, 50)
            progress_bar.progress(pct, text=f"Active Agent: {node.replace('_', ' ').title()}...")

            # Small pause to allow visual animation transition
            time.sleep(0.3)

        progress_bar.progress(100, text="All agents have finished execution!")
        time.sleep(0.5)
        progress_bar.empty()

        st.session_state.active_node = "finalize"
        st.session_state.completed_nodes.add("finalize")
        stepper_placeholder.markdown(
            render_pipeline_stepper(None, st.session_state.completed_nodes),
            unsafe_allow_html=True,
        )

        st.session_state.cleaned_output_path = output_clean_path
        st.session_state.workflow_finished = True
        st.session_state.is_running = False
        st.balloons()
        st.rerun()

    except Exception as exc:
        st.session_state.is_running = False
        st.error(f"❌ Execution encountered an error: {str(exc)}")
        progress_bar.empty()

# ============================================================
# AGENT STEP OUTPUT INSPECTOR
# ============================================================
if st.session_state.node_outputs:
    st.markdown("---")
    st.subheader("🕵️ Step-by-Step Agent Outputs & Reasoning")

    tabs = st.tabs(
        [
            "1. 📊 Profiler",
            "2. 🧠 Understanding Agent",
            "3. 💻 Cleaning Agent",
            "4. ⚙️ Execution Engine",
            "5. 🛡️ Validation Auditor",
            "6. 🏁 Final Summary",
        ]
    )

    # Tab 1: Profiler
    with tabs[0]:
        if "profile_original" in st.session_state.node_outputs:
            summary = st.session_state.node_outputs["profile_original"].get("original_summary", "")
            st.markdown(
                """
                <div class="agent-card">
                    <div class="agent-header">
                        <span class="agent-title">📊 Dataset Profiler Output</span>
                        <span class="agent-badge badge-pass">STRUCTURE DETECTED</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("View Full Context-Budgeted Profile", expanded=True):
                st.text(summary)
        else:
            st.info("Profiler step has not run yet.")

    # Tab 2: Understanding Agent
    with tabs[1]:
        if "understanding_agent" in st.session_state.node_outputs:
            u_out = st.session_state.node_outputs["understanding_agent"]
            u_report = u_out.get("understanding_report", "")
            cleaning_req = u_out.get("cleaning_required", True)

            status_badge = (
                '<span class="agent-badge badge-fail">CLEANING REQUIRED: YES</span>'
                if cleaning_req
                else '<span class="agent-badge badge-pass">CLEANING REQUIRED: NO</span>'
            )

            st.markdown(
                f"""
                <div class="agent-card">
                    <div class="agent-header">
                        <span class="agent-title">🧠 Understanding Agent Diagnosis</span>
                        {status_badge}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(u_report)
        else:
            st.info("Understanding Agent has not run yet.")

    # Tab 3: Cleaning Agent
    with tabs[2]:
        if "cleaning_agent" in st.session_state.node_outputs:
            c_out = st.session_state.node_outputs["cleaning_agent"]
            code = c_out.get("cleaning_code", "")

            st.markdown(
                """
                <div class="agent-card">
                    <div class="agent-header">
                        <span class="agent-title">💻 Cleaning Agent Generated Python Script</span>
                        <span class="agent-badge badge-working">PANDAS CODE READY</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.code(code, language="python")
            st.download_button(
                "📥 Download Cleaning Python Script",
                data=code,
                file_name="cleaning_script.py",
                mime="text/x-python",
            )
        else:
            st.info("Cleaning Agent has not run yet.")

    # Tab 4: Execution Engine
    with tabs[3]:
        if "execute_cleaning" in st.session_state.node_outputs:
            e_out = st.session_state.node_outputs["execute_cleaning"]
            success = e_out.get("execution_success", False)
            error = e_out.get("execution_error", "")
            history = e_out.get("cleaning_history", [])

            badge = (
                '<span class="agent-badge badge-pass">EXECUTION SUCCESS</span>'
                if success
                else '<span class="agent-badge badge-fail">EXECUTION FAILED</span>'
            )

            st.markdown(
                f"""
                <div class="agent-card">
                    <div class="agent-header">
                        <span class="agent-title">⚙️ Code Execution Results</span>
                        {badge}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if success:
                st.success("✅ The Python cleaning script ran with zero exceptions and outputted cleaned_output.csv.")
            else:
                st.error(f"Execution Error: {error}")

            if history:
                st.markdown("**Execution History Trail:**")
                for item in history:
                    st.markdown(f"- `{item}`")
        else:
            st.info("Code Execution has not run yet.")

    # Tab 5: Validation Auditor
    with tabs[4]:
        if "validation_agent" in st.session_state.node_outputs:
            v_out = st.session_state.node_outputs["validation_agent"]
            v_report = v_out.get("validation_report", "")
            passed = v_out.get("validation_passed", False)
            blocked = v_out.get("blocked", False)

            if passed:
                v_badge = '<span class="agent-badge badge-pass">VALIDATION: PASS</span>'
            elif blocked:
                v_badge = '<span class="agent-badge badge-blocked">VALIDATION: BLOCKED</span>'
            else:
                v_badge = '<span class="agent-badge badge-fail">VALIDATION: FAIL (LOOPING BACK)</span>'

            st.markdown(
                f"""
                <div class="agent-card">
                    <div class="agent-header">
                        <span class="agent-title">🛡️ Validation Auditor Report</span>
                        {v_badge}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(v_report)
        else:
            st.info("Validation Auditor has not run yet.")

    # Tab 6: Final Summary
    with tabs[5]:
        if "finalize" in st.session_state.node_outputs:
            f_out = st.session_state.node_outputs["finalize"]
            messages = f_out.get("messages", [])
            final_text = messages[-1].content if messages else "Completed."
            st.markdown(
                """
                <div class="agent-card">
                    <div class="agent-header">
                        <span class="agent-title">🏁 Workflow Conclusion</span>
                        <span class="agent-badge badge-pass">FINAL REPORT</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(final_text)
        else:
            st.info("Final node has not run yet.")

# ============================================================
# BEFORE VS. AFTER CSV COMPARISON SECTION
# ============================================================
cleaned_path = st.session_state.get("cleaned_output_path")

if cleaned_path and Path(cleaned_path).exists():
    cleaned_df = pd.read_csv(cleaned_path)

    st.markdown("---")
    st.subheader("✨ 3. Before vs. After Data Comparison")

    clean_rows, clean_cols = cleaned_df.shape
    clean_missing = int(cleaned_df.isna().sum().sum())
    clean_duplicates = int(cleaned_df.duplicated().sum())

    # Delta Metrics Row
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        row_delta = clean_rows - raw_rows
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-val">{clean_rows:,}</div>
                <div class="metric-lbl">Cleaned Rows</div>
                <div class="metric-delta {'delta-positive' if row_delta <= 0 else 'delta-negative'}">
                    {'Preserved' if row_delta == 0 else f'{row_delta} rows (removed invalid/duplicates)'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        col_delta = clean_cols - raw_cols
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-val">{clean_cols}</div>
                <div class="metric-lbl">Cleaned Columns</div>
                <div class="metric-delta {'delta-positive' if col_delta <= 0 else 'delta-negative'}">
                    {'Same columns' if col_delta == 0 else f'{col_delta} columns adjusted'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        missing_diff = clean_missing - raw_missing
        pct_improvement = (
            ((raw_missing - clean_missing) / raw_missing * 100) if raw_missing > 0 else 100
        )
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-val" style="color: {'#10b981' if clean_missing == 0 else '#f59e0b'};">{clean_missing:,}</div>
                <div class="metric-lbl">Missing Values After</div>
                <div class="metric-delta delta-positive">
                    {pct_improvement:.0f}% Missingness Eliminated (Δ {missing_diff})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m4:
        dup_diff = clean_duplicates - raw_duplicates
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-val" style="color: {'#10b981' if clean_duplicates == 0 else '#ef4444'};">{clean_duplicates:,}</div>
                <div class="metric-lbl">Duplicates After</div>
                <div class="metric-delta delta-positive">
                    {'Zero Duplicates Left!' if clean_duplicates == 0 else f'Δ {dup_diff}'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Side by side dataset tabs
    tab_before, tab_after, tab_diff = st.tabs(
        ["🔴 Raw Dataset (Before)", "🟢 Cleaned Dataset (After)", "⚖️ Column Quality Matrix"]
    )

    with tab_before:
        st.caption("Original uncleaned data:")
        st.dataframe(raw_df, use_container_width=True, height=280)

    with tab_after:
        st.caption("Autonomous multi-agent cleaned dataset:")
        st.dataframe(cleaned_df, use_container_width=True, height=280)

    with tab_diff:
        # Build comparative summary
        all_cols = sorted(list(set(raw_df.columns).union(set(cleaned_df.columns))))
        matrix_rows = []
        for col in all_cols:
            raw_has = col in raw_df.columns
            clean_has = col in cleaned_df.columns

            raw_dt = str(raw_df[col].dtype) if raw_has else "N/A (Dropped)"
            clean_dt = str(cleaned_df[col].dtype) if clean_has else "N/A (Dropped)"

            raw_nulls = int(raw_df[col].isna().sum()) if raw_has else 0
            clean_nulls = int(cleaned_df[col].isna().sum()) if clean_has else 0

            status = "Preserved & Cleaned"
            if not clean_has:
                status = "Dropped (Constant/Invalid)"
            elif raw_nulls > 0 and clean_nulls == 0:
                status = "Imputed / Cleaned (100% Fixed)"
            elif raw_dt != clean_dt:
                status = f"Type Converted ({raw_dt} ➔ {clean_dt})"

            matrix_rows.append(
                {
                    "Column": col,
                    "Raw Type": raw_dt,
                    "Cleaned Type": clean_dt,
                    "Raw Nulls": raw_nulls,
                    "Cleaned Nulls": clean_nulls,
                    "Transformation Status": status,
                }
            )

        matrix_df = pd.DataFrame(matrix_rows)
        st.dataframe(matrix_df, use_container_width=True)

    # Download Cleaned CSV Button
    st.markdown("---")
    csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
    d1, d2 = st.columns([1, 1])
    with d1:
        st.download_button(
            label="📥 Download Cleaned CSV File",
            data=csv_bytes,
            file_name="cleaned_output.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )
    with d2:
        # Build comprehensive audit log markdown
        audit_text = f"""# Multi-Agent Autonomous Data Cleaning Audit Report

## Dataset Summary
- **Original Dataset**: {raw_csv_path}
- **Original Shape**: {raw_df.shape} ({raw_missing} missing values, {raw_duplicates} duplicates)
- **Cleaned Shape**: {cleaned_df.shape} ({clean_missing} missing values, {clean_duplicates} duplicates)

## Column Transformations
{matrix_df.to_markdown(index=False)}

## Final Validator Status
{st.session_state.node_outputs.get('validation_agent', {}).get('validation_report', 'N/A')}
"""
        st.download_button(
            label="📋 Download Full Audit Report (.md)",
            data=audit_text,
            file_name="cleaning_audit_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
