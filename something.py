import os
import re
import json
import shutil
from pathlib import Path
from typing import TypedDict, Annotated, Optional

import pandas as pd
import numpy as np

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Example:
# set GROQ_API_KEY=your_key_here       # Windows CMD
# $env:GROQ_API_KEY="your_key_here"    # PowerShell
#
# NEVER hard-code API keys into this file.

MODEL = "openai/gpt-oss-120b"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY environment variable is missing."
    )

llm = ChatGroq(
    model=MODEL,
    temperature=0,
    api_key=GROQ_API_KEY,
)


# ============================================================
# 2. DATASET PROFILER
# ============================================================


def summarize_csv(
    path: str,
    max_output_chars: int = 30000,
    top_n_categories: int = 5,
    max_examples: int = 3,
    max_correlation_pairs: int = 30,
    string_preview_length: int = 80,
) -> str:
    """
    Context-efficient dataset profiler for LLM-based data cleaning.

    Design goals:
    - Preserve important data-quality information.
    - Never dump all categorical values.
    - Never dump large raw datasets.
    - Limit long strings.
    - Report only meaningful correlations.
    - Detect suspicious numeric/date columns.
    - Provide compact representative examples.
    - Enforce a hard output-size budget.
    """

    df = pd.read_csv(path)

    lines = []

    # ============================================================
    # Helper functions
    # ============================================================

    def safe_str(value, max_len=string_preview_length):
        """Convert a value to a compact string."""
        if pd.isna(value):
            return "<NULL>"

        value = str(value)

        if len(value) > max_len:
            return value[:max_len] + "... [truncated]"

        return value

    def add(section):
        """Append a section."""
        lines.append(section)

    # ============================================================
    # BASIC DATASET INFORMATION
    # ============================================================

    add("=== DATASET OVERVIEW ===")

    add(f"Rows: {len(df):,}")
    add(f"Columns: {len(df.columns):,}")

    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    add(f"Memory: {memory_mb:.2f} MB")

    add(
        "Columns: "
        + ", ".join(str(c) for c in df.columns)
    )

    # ============================================================
    # COLUMN PROFILE
    # ============================================================

    add("\n=== COLUMN PROFILE ===")

    profile_rows = []

    for col in df.columns:

        s = df[col]

        missing = int(s.isna().sum())
        missing_pct = (
            missing / len(df) * 100
            if len(df)
            else 0
        )

        unique = int(s.nunique(dropna=True))

        unique_pct = (
            unique / len(df) * 100
            if len(df)
            else 0
        )

        dtype = str(s.dtype)

        profile_rows.append(
            {
                "column": str(col),
                "dtype": dtype,
                "missing": missing,
                "missing_%": round(missing_pct, 2),
                "unique": unique,
                "unique_%": round(unique_pct, 2),
            }
        )

    profile_df = pd.DataFrame(profile_rows)

    add(profile_df.to_string(index=False))

    # ============================================================
    # DUPLICATES
    # ============================================================

    duplicate_count = int(df.duplicated().sum())

    duplicate_pct = (
        duplicate_count / len(df) * 100
        if len(df)
        else 0
    )

    add("\n=== DUPLICATES ===")

    add(
        f"Duplicate rows: {duplicate_count:,} "
        f"({duplicate_pct:.2f}%)"
    )

    # ============================================================
    # CONSTANT COLUMNS
    # ============================================================

    constant_cols = [
        str(c)
        for c in df.columns
        if df[c].nunique(dropna=False) <= 1
    ]

    add("\n=== CONSTANT COLUMNS ===")

    add(
        ", ".join(constant_cols)
        if constant_cols
        else "None"
    )

    # ============================================================
    # NUMERIC COLUMNS
    # ============================================================

    num_df = df.select_dtypes(include=np.number)

    add("\n=== NUMERIC COLUMN ANALYSIS ===")

    if num_df.empty:

        add("No numeric columns.")

    else:

        numeric_rows = []

        for col in num_df.columns:

            s = num_df[col]

            valid = s.dropna()

            if valid.empty:
                continue

            q1 = valid.quantile(0.25)
            median = valid.median()
            q3 = valid.quantile(0.75)

            iqr = q3 - q1

            if pd.isna(iqr) or iqr == 0:

                outliers = 0

            else:

                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                outliers = int(
                    ((valid < lower) | (valid > upper)).sum()
                )

            outlier_pct = (
                outliers / len(df) * 100
                if len(df)
                else 0
            )

            numeric_rows.append(
                {
                    "column": str(col),
                    "count": len(valid),
                    "missing": int(s.isna().sum()),
                    "mean": round(valid.mean(), 4),
                    "median": round(median, 4),
                    "std": round(valid.std(), 4),
                    "min": round(valid.min(), 4),
                    "q1": round(q1, 4),
                    "q3": round(q3, 4),
                    "max": round(valid.max(), 4),
                    "skew": round(valid.skew(), 4),
                    "kurtosis": round(valid.kurtosis(), 4),
                    "outliers": outliers,
                    "outlier_%": round(outlier_pct, 2),
                }
            )

        numeric_summary = pd.DataFrame(numeric_rows)

        add(
            numeric_summary.to_string(index=False)
        )

    # ============================================================
    # STRONG CORRELATIONS ONLY
    # ============================================================

    add("\n=== STRONG NUMERIC CORRELATIONS ===")

    if num_df.shape[1] >= 2:

        corr = num_df.corr()

        pairs = []

        for i, col1 in enumerate(corr.columns):

            for j, col2 in enumerate(corr.columns):

                if j <= i:
                    continue

                value = corr.loc[col1, col2]

                if pd.isna(value):
                    continue

                if abs(value) >= 0.70:

                    pairs.append(
                        (
                            abs(value),
                            str(col1),
                            str(col2),
                            float(value),
                        )
                    )

        pairs.sort(reverse=True)

        if pairs:

            for _, col1, col2, value in pairs[
                :max_correlation_pairs
            ]:

                add(
                    f"{col1} <-> {col2}: "
                    f"{value:.3f}"
                )

        else:

            add("No strong correlations found (|r| >= 0.70).")

    else:

        add("Not enough numeric columns.")

    # ============================================================
    # CATEGORICAL COLUMNS
    # ============================================================

    cat_df = df.select_dtypes(
        include=["object", "category"]
    )

    add("\n=== CATEGORICAL COLUMN ANALYSIS ===")

    if cat_df.empty:

        add("No categorical columns.")

    else:

        for col in cat_df.columns:

            s = cat_df[col]

            non_null = s.dropna()

            unique_count = int(
                non_null.nunique()
            )

            unique_ratio = (
                unique_count / len(df)
                if len(df)
                else 0
            )

            missing_count = int(s.isna().sum())

            missing_pct = (
                missing_count / len(df) * 100
                if len(df)
                else 0
            )

            # String lengths
            if len(non_null) > 0:

                lengths = (
                    non_null.astype(str)
                    .str.len()
                )

                avg_len = lengths.mean()
                max_len = lengths.max()

            else:

                avg_len = 0
                max_len = 0

            add(f"\nColumn: {col}")

            add(
                f"  Unique: {unique_count:,} "
                f"({unique_ratio * 100:.2f}% of rows)"
            )

            add(
                f"  Missing: {missing_count:,} "
                f"({missing_pct:.2f}%)"
            )

            add(
                f"  High cardinality: "
                f"{'YES' if unique_ratio > 0.50 else 'NO'}"
            )

            add(
                f"  Avg string length: {avg_len:.1f}"
            )

            add(
                f"  Max string length: {max_len:,}"
            )

            # ----------------------------------------------------
            # Important:
            # Only top values, NEVER all unique values.
            # ----------------------------------------------------

            if unique_count <= 20:

                top_values = (
                    non_null
                    .value_counts()
                    .head(top_n_categories)
                )

            else:

                top_values = (
                    non_null
                    .value_counts()
                    .head(top_n_categories)
                )

            if not top_values.empty:

                add("  Most frequent values:")

                for value, count in top_values.items():

                    add(
                        f"    {safe_str(value)} "
                        f"({count:,})"
                    )

    # ============================================================
    # MIXED TYPES
    # ============================================================

    add("\n=== MIXED-TYPE COLUMNS ===")

    mixed_cols = []

    for col in df.columns:

        non_null = df[col].dropna()

        if len(non_null) == 0:
            continue

        type_count = (
            non_null
            .map(type)
            .nunique()
        )

        if type_count > 1:

            mixed_cols.append(
                f"{col} "
                f"({type_count} Python types)"
            )

    add(
        "\n".join(mixed_cols)
        if mixed_cols
        else "None detected."
    )

    # ============================================================
    # NUMERIC-LOOKING STRINGS
    # ============================================================

    add(
        "\n=== POSSIBLE NUMERIC COLUMNS STORED AS TEXT ==="
    )

    suspicious_numeric = []

    for col in cat_df.columns:

        s = cat_df[col].dropna()

        if len(s) == 0:
            continue

        converted = pd.to_numeric(
            s.astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )

        ratio = converted.notna().mean()

        if ratio >= 0.90:

            suspicious_numeric.append(
                f"{col}: "
                f"{ratio * 100:.1f}% numeric-convertible"
            )

    add(
        "\n".join(suspicious_numeric)
        if suspicious_numeric
        else "None detected."
    )

    # ============================================================
    # DATE-LOOKING STRINGS
    # ============================================================

    add(
        "\n=== POSSIBLE DATE COLUMNS STORED AS TEXT ==="
    )

    suspicious_dates = []

    for col in cat_df.columns:

        s = cat_df[col].dropna()

        if len(s) == 0:
            continue

        try:

            converted = pd.to_datetime(
                s.astype(str),
                errors="coerce",
            )

            ratio = converted.notna().mean()

            if ratio >= 0.90:

                suspicious_dates.append(
                    f"{col}: "
                    f"{ratio * 100:.1f}% date-convertible"
                )

        except Exception:
            pass

    add(
        "\n".join(suspicious_dates)
        if suspicious_dates
        else "None detected."
    )

    # ============================================================
    # REPRESENTATIVE DATA SAMPLE
    # ============================================================

    add("\n=== REPRESENTATIVE DATA SAMPLE ===")

    if len(df) > 0:

        # Small sample only.
        sample_size = min(
            max_examples,
            len(df)
        )

        sample = df.sample(
            n=sample_size,
            random_state=42,
        ).copy()

        # Truncate every value before sending it.
        for col in sample.columns:

            sample[col] = sample[col].map(
                lambda x: safe_str(x)
            )

        add(
            sample.to_string(index=False)
        )

    else:

        add("Dataset is empty.")

    # ============================================================
    # OUTPUT SIZE CONTROL
    # ============================================================

    result = "\n".join(lines)

    if len(result) > max_output_chars:

        result = (
            result[:max_output_chars]
            + "\n\n"
            + "[PROFILE TRUNCATED: "
            + f"output exceeded {max_output_chars:,} characters]"
        )

    return result


# ============================================================
# 3. UNDERSTANDING AGENT
# ============================================================

UNDERSTANDING_SYSTEM_PROMPT = """
You are an expert data scientist responsible for understanding
a dataset before any cleaning is performed.

You will receive a dataset profile containing:
- shape
- column names
- dtypes
- missing values
- duplicates
- constant columns
- numeric statistics
- outlier counts
- categorical values
- possible mixed types
- possible numeric/date columns
- sample rows

Your job is to decide whether cleaning is actually required.

Analyze:
1. Missing values
2. Duplicate rows
3. Incorrect dtypes
4. Numeric-looking strings
5. Date-looking strings
6. Mixed types
7. Constant/unnecessary columns
8. Important outliers
9. Inconsistent or suspicious categorical values
10. Any other important data-integrity problem

IMPORTANT:
- Do NOT generate Python code.
- Do NOT recommend cleaning merely because an IQR outlier exists.
- Do NOT remove valid categories just because they are uncommon.
- Do NOT recommend dropping rows unless there is a strong data-quality reason.
- Preserve valid information.
- Be conservative.
- If the dataset is already sufficiently clean, say cleaning is not needed.

At the beginning of your response, write exactly:

CLEANING_REQUIRED: YES

or

CLEANING_REQUIRED: NO

Then provide a concise but complete analysis.

If cleaning is required, clearly explain the exact columns
and problems that the cleaning agent should address.
"""


understanding_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", UNDERSTANDING_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

understanding_chain = understanding_prompt | llm


# ============================================================
# 4. CLEANING AGENT
# ============================================================

CLEANING_SYSTEM_PROMPT = """
You are an expert Python data engineer.

You will receive:
1. The CURRENT dataset summary
2. The ORIGINAL dataset summary
3. The original data-quality analysis
4. Validation feedback from the previous cleaning attempt
5. Cleaning history

Generate a complete ready-to-run pandas Python script.

The script must load the CURRENT dataset, clean it according
to the identified problems, and save the result to:

cleaned_output.csv

IMPORTANT DATA-INTEGRITY RULES:

1. NEVER intentionally remove valid categorical values.
   Example:
   If a categorical column originally contains:
   Single, In Relationship, Complicated
   you must NOT remove "Complicated" merely because it is less frequent.

2. NEVER filter rows unless the analysis/validation clearly
   identifies those rows as invalid or duplicates.

3. Do not remove an entire category just because it is rare.

4. Preserve the original columns unless dropping a column is
   explicitly justified.

5. Missing values must be handled according to the meaning
   and dtype of each column.

6. Convert dtypes only when the data clearly supports it.

7. Do not convert a genuine categorical column into numeric
   simply because some values look numeric.

8. Do not treat every statistical outlier as an invalid value.
   Only handle outliers when the data-quality analysis
   supports doing so.

9. Preserve row information whenever possible.

10. Never introduce new missing values accidentally.

11. Never silently rename columns.

12. Never overwrite valid values with arbitrary defaults.

13. If validation feedback says a previous cleaning step
    caused data loss, fix that exact problem first.

14. The CURRENT dataset is the input for this iteration.
    Do not go back to the original dataset unless explicitly
    required.

15. At the end, the script MUST execute:
    df.to_csv("cleaned_output.csv", index=False)

16. Add a short comment before every important cleaning step.

17. Output ONLY Python code inside one code block.
    No explanation outside the code.

The generated code may assume:

df = pd.read_csv("<CURRENT_DATASET_PATH>")
"""


cleaning_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CLEANING_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

cleaning_chain = cleaning_prompt | llm


# ============================================================
# 5. VALIDATION AGENT
# ============================================================

VALIDATION_SYSTEM_PROMPT = """
You are a strict data quality auditor.

You will receive:
1. Original dataset summary
2. Current dataset summary
3. Initial understanding analysis
4. Cleaning history
5. Previous validation feedback

Determine whether the CURRENT dataset is sufficiently clean.

Compare the original dataset with the current dataset.

Check:

- Missing values
- Duplicate rows
- Correct dtypes
- Numeric-looking strings
- Date-looking columns
- Mixed-type columns
- Constant/unnecessary columns
- Important outliers
- Unexpected row loss
- Unexpected column loss
- Unexpected category/value loss
- New missing values
- New data-integrity problems
- Whether previous validation problems were actually fixed

VERY IMPORTANT:

A valid categorical value disappearing is a data-integrity
problem.

For example, if the original dataset has:
Relationship_Status:
- Single
- In Relationship
- Complicated

and the current dataset only has:
- Single
- In Relationship

then the cleaning is NOT successful unless there is explicit
evidence that "Complicated" is invalid.

Do not demand that every statistical outlier disappear.
Do not demand changes that are not justified by the original
data-quality analysis.

If important problems remain, cleaning is needed again.

If the dataset is sufficiently clean, validation passes.

Respond EXACTLY in this format:

VALIDATION_STATUS: PASS

or

VALIDATION_STATUS: FAIL

or

VALIDATION_STATUS: BLOCKED

REASON:
<short explanation>

REMAINING_ISSUES:
- <issue 1>
- <issue 2>
- <issue 3>

If there are no issues:

REMAINING_ISSUES:
- None

Use BLOCKED only when the problem cannot be safely resolved
automatically or the execution result indicates that the
workflow cannot continue safely.
"""


validation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", VALIDATION_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

validation_chain = validation_prompt | llm


# ============================================================
# 6. LANGGRAPH STATE
# ============================================================

class AgentState(TypedDict, total=False):

    # LangGraph conversation messages
    messages: Annotated[list, add_messages]

    # Paths
    original_dataset_path: str
    current_dataset_path: str
    output_path: str

    # Profiles
    original_summary: str
    current_summary: str

    # Agent reports
    understanding_report: str
    validation_report: str

    # Decisions
    cleaning_required: bool
    validation_passed: bool
    blocked: bool

    # Generated code
    cleaning_code: str

    # Execution
    execution_success: bool
    execution_error: str

    # Number of cleaning cycles
    iteration: int

    # History
    cleaning_history: list


# ============================================================
# 7. HELPERS
# ============================================================

def extract_code(text: str) -> str:
    """
    Extract Python code from a model response.
    """

    match = re.search(
        r"```python\s*(.*?)```",
        text,
        re.DOTALL | re.IGNORECASE,
    )

    if match:
        return match.group(1).strip()

    match = re.search(
        r"```\s*(.*?)```",
        text,
        re.DOTALL,
    )

    if match:
        return match.group(1).strip()

    return text.strip()


def parse_cleaning_required(text: str) -> Optional[bool]:

    match = re.search(
        r"CLEANING_REQUIRED\s*:\s*(YES|NO)",
        text,
        re.IGNORECASE,
    )

    if not match:
        return None

    return match.group(1).upper() == "YES"


def parse_validation_status(text: str) -> Optional[str]:

    match = re.search(
        r"VALIDATION_STATUS\s*:\s*(PASS|FAIL|BLOCKED)",
        text,
        re.IGNORECASE,
    )

    if not match:
        return None

    return match.group(1).upper()


def replace_input_path(code: str, dataset_path: str) -> str:
    """
    Replace the placeholder used by the cleaning agent.

    Also handles a few common variations.
    """

    replacements = [
        'pd.read_csv("<CURRENT_DATASET_PATH>")',
        "pd.read_csv('<CURRENT_DATASET_PATH>')",
        'pd.read_csv("<path>")',
        "pd.read_csv('<path>')",
    ]

    escaped_path = dataset_path.replace("\\", "\\\\")

    for old in replacements:
        code = code.replace(
            old,
            f'pd.read_csv(r"{escaped_path}")',
        )

    return code


def ensure_output_exists() -> bool:
    return Path("cleaned_output.csv").exists()


# ============================================================
# 8. NODE: PROFILE ORIGINAL DATASET
# ============================================================

def profile_original(state: AgentState):

    path = state["original_dataset_path"]

    summary = summarize_csv(path)

    return {
        "original_summary": summary,
        "current_summary": summary,
        "current_dataset_path": path,
        "iteration": 0,
        "cleaning_history": [],
    }


# ============================================================
# 9. NODE: UNDERSTANDING AGENT
# ============================================================

def understanding_agent(state: AgentState):

    prompt_text = f"""
DATASET SUMMARY:

{state["original_summary"]}

Analyze this dataset and decide whether cleaning is required.
Remember: do not generate Python code.
"""

    response = understanding_chain.invoke(
        {
            "messages": [
                HumanMessage(content=prompt_text)
            ]
        }
    )

    report = response.content

    decision = parse_cleaning_required(report)

    # If the model does not follow the exact format,
    # fail safely instead of guessing.
    if decision is None:

        raise RuntimeError(
            "Understanding agent did not return "
            "CLEANING_REQUIRED: YES/NO.\n\n"
            f"Response:\n{report}"
        )

    return {
        "understanding_report": report,
        "cleaning_required": decision,
        "messages": [response],
    }


# ============================================================
# 10. ROUTER AFTER UNDERSTANDING
# ============================================================

def route_after_understanding(state: AgentState):

    if state["cleaning_required"]:
        return "cleaning_agent"

    return "finalize"


# ============================================================
# 11. NODE: CLEANING AGENT
# ============================================================

def cleaning_agent(state: AgentState):

    previous_validation = state.get(
        "validation_report",
        "No previous validation exists. This is the first cleaning attempt.",
    )

    history = state.get(
        "cleaning_history",
        [],
    )

    history_text = "\n\n".join(
        history[-5:]
    ) if history else "No previous cleaning history."

    prompt_text = f"""
CURRENT DATASET PATH:
{state["current_dataset_path"]}

CURRENT DATASET SUMMARY:
{state["current_summary"]}

ORIGINAL DATASET SUMMARY:
{state["original_summary"]}

INITIAL DATA QUALITY ANALYSIS:
{state["understanding_report"]}

PREVIOUS VALIDATION FEEDBACK:
{previous_validation}

CLEANING HISTORY:
{history_text}

This is cleaning iteration:
{state.get("iteration", 0) + 1}

Generate the corrected pandas cleaning script.

Remember:
- Work on the CURRENT dataset.
- Preserve valid information.
- Fix the validation issues.
- Save the final result to cleaned_output.csv.
"""

    response = cleaning_chain.invoke(
        {
            "messages": [
                HumanMessage(content=prompt_text)
            ]
        }
    )

    code = extract_code(response.content)

    return {
        "cleaning_code": code,
        "messages": [response],
    }


# ============================================================
# 12. NODE: EXECUTE CLEANING CODE
# ============================================================

def execute_cleaning(state: AgentState):

    code = state["cleaning_code"]
    current_path = state["current_dataset_path"]

    # Remove a previous output so a failed execution cannot
    # accidentally reuse an old output file.
    output_path = Path("cleaned_output.csv")

    if output_path.exists():
        output_path.unlink()

    code_to_run = replace_input_path(
        code,
        current_path,
    )

    exec_globals = {
        "pd": pd,
        "np": np,
        "os": os,
        "Path": Path,
        "__name__": "__main__",
    }

    exec_locals = {}

    try:

        exec(
            code_to_run,
            exec_globals,
            exec_locals,
        )

        if not output_path.exists():

            raise RuntimeError(
                "Cleaning code executed without an exception, "
                "but cleaned_output.csv was not created."
            )

        # Validate that pandas can actually read the result.
        cleaned_df = pd.read_csv(
            output_path
        )

        if cleaned_df.empty:

            raise RuntimeError(
                "Cleaning produced an empty dataset."
            )

        history_entry = (
            f"Iteration {state.get('iteration', 0) + 1}: "
            "Cleaning code executed successfully. "
            f"Output shape: {cleaned_df.shape}."
        )

        return {
            "execution_success": True,
            "execution_error": "",
            "output_path": str(
                output_path.resolve()
            ),
            "current_dataset_path": str(
                output_path.resolve()
            ),
            "iteration": state.get(
                "iteration",
                0,
            ) + 1,
            "cleaning_history": (
                state.get("cleaning_history", [])
                + [history_entry]
            ),
            "messages": [
                HumanMessage(
                    content=history_entry
                )
            ],
        }

    except Exception as exc:

        error_text = (
            f"Cleaning execution failed: "
            f"{type(exc).__name__}: {exc}"
        )

        history_entry = (
            f"Iteration {state.get('iteration', 0) + 1}: "
            f"{error_text}"
        )

        return {
            "execution_success": False,
            "execution_error": error_text,
            "iteration": state.get(
                "iteration",
                0,
            ) + 1,
            "cleaning_history": (
                state.get("cleaning_history", [])
                + [history_entry]
            ),
            "messages": [
                HumanMessage(
                    content=error_text
                )
            ],
        }


# ============================================================
# 13. ROUTER AFTER EXECUTION
# ============================================================

def route_after_execution(state: AgentState):

    if state["execution_success"]:
        return "profile_cleaned"

    # Execution errors go back to the cleaning agent.
    # There is deliberately NO retry counter here.
    return "cleaning_agent"


# ============================================================
# 14. NODE: PROFILE CLEANED DATASET
# ============================================================

def profile_cleaned(state: AgentState):

    current_path = state["current_dataset_path"]

    summary = summarize_csv(
        current_path
    )

    return {
        "current_summary": summary,
    }


# ============================================================
# 15. NODE: VALIDATION AGENT
# ============================================================

def validation_agent(state: AgentState):

    history = state.get(
        "cleaning_history",
        [],
    )

    history_text = "\n".join(history)

    prompt_text = f"""
ORIGINAL DATASET SUMMARY:
{state["original_summary"]}

CURRENT DATASET SUMMARY:
{state["current_summary"]}

INITIAL UNDERSTANDING:
{state["understanding_report"]}

CLEANING HISTORY:
{history_text}

PREVIOUS VALIDATION:
{state.get("validation_report", "None")}

CURRENT EXECUTION STATUS:
{state.get("execution_success", False)}

CURRENT EXECUTION ERROR:
{state.get("execution_error", "")}

Audit the current dataset.

Compare original vs current carefully.
Pay special attention to:
- lost rows
- lost columns
- lost categorical values
- new missing values
- duplicates
- dtype problems
- invalid conversions
- unjustified outlier removal
- unresolved original problems
- newly introduced problems
"""

    response = validation_chain.invoke(
        {
            "messages": [
                HumanMessage(content=prompt_text)
            ]
        }
    )

    report = response.content

    status = parse_validation_status(
        report
    )

    if status is None:

        raise RuntimeError(
            "Validation agent did not return "
            "VALIDATION_STATUS: PASS/FAIL/BLOCKED.\n\n"
            f"Response:\n{report}"
        )

    return {
        "validation_report": report,
        "validation_passed": (
            status == "PASS"
        ),
        "blocked": (
            status == "BLOCKED"
        ),
        "messages": [response],
    }


# ============================================================
# 16. ROUTER AFTER VALIDATION
# ============================================================

def route_after_validation(state: AgentState):

    if state.get("blocked", False):
        return "finalize"

    if state.get("validation_passed", False):
        return "finalize"

    # Validation failed:
    # send the feedback back to the cleaning agent.
    return "cleaning_agent"


# ============================================================
# 17. FINAL NODE
# ============================================================

def finalize(state: AgentState):

    current_path = state.get(
        "current_dataset_path"
    )

    final_message = ""

    if state.get("blocked", False):

        final_message = (
            "WORKFLOW BLOCKED.\n\n"
            "The validator determined that the dataset "
            "cannot be safely cleaned automatically.\n\n"
            f"Final dataset: {current_path}\n\n"
            f"Validation report:\n"
            f"{state.get('validation_report', '')}"
        )

    elif state.get("validation_passed", False):

        final_message = (
            "WORKFLOW COMPLETED SUCCESSFULLY.\n\n"
            f"Final cleaned dataset:\n"
            f"{current_path}\n\n"
            f"Cleaning iterations: "
            f"{state.get('iteration', 0)}\n\n"
            "Final validation:\n"
            f"{state.get('validation_report', '')}"
        )

    else:

        # This branch means the understanding agent decided
        # that cleaning was not required.
        final_message = (
            "WORKFLOW COMPLETED.\n\n"
            "The understanding agent determined that "
            "cleaning was not required.\n\n"
            f"Dataset:\n{current_path}\n\n"
            "Analysis:\n"
            f"{state.get('understanding_report', '')}"
        )

    return {
        "messages": [
            HumanMessage(
                content=final_message
            )
        ]
    }


# ============================================================
# 18. BUILD LANGGRAPH
# ============================================================

builder = StateGraph(
    AgentState
)

builder.add_node(
    "profile_original",
    profile_original,
)

builder.add_node(
    "understanding_agent",
    understanding_agent,
)

builder.add_node(
    "cleaning_agent",
    cleaning_agent,
)

builder.add_node(
    "execute_cleaning",
    execute_cleaning,
)

builder.add_node(
    "profile_cleaned",
    profile_cleaned,
)

builder.add_node(
    "validation_agent",
    validation_agent,
)

builder.add_node(
    "finalize",
    finalize,
)


# Entry
builder.add_edge(
    START,
    "profile_original",
)

# Original profile -> understanding
builder.add_edge(
    "profile_original",
    "understanding_agent",
)

# Understanding -> cleaning OR final
builder.add_conditional_edges(
    "understanding_agent",
    route_after_understanding,
    {
        "cleaning_agent": "cleaning_agent",
        "finalize": "finalize",
    },
)

# Cleaning -> execution
builder.add_edge(
    "cleaning_agent",
    "execute_cleaning",
)

# Execution -> profile cleaned OR cleaning again
builder.add_conditional_edges(
    "execute_cleaning",
    route_after_execution,
    {
        "profile_cleaned": "profile_cleaned",
        "cleaning_agent": "cleaning_agent",
    },
)

# Profile -> validation
builder.add_edge(
    "profile_cleaned",
    "validation_agent",
)

# Validation -> final OR cleaning again
builder.add_conditional_edges(
    "validation_agent",
    route_after_validation,
    {
        "finalize": "finalize",
        "cleaning_agent": "cleaning_agent",
    },
)

# Final
builder.add_edge(
    "finalize",
    END,
)

graph = builder.compile()


# ============================================================
# 19. RUN WORKFLOW
# ============================================================

def run_data_cleaning(
    csv_path: str,
):
    """
    Run the complete autonomous workflow.

    There is no application-level MAX_ITERATIONS.

    The graph continues:
        cleaning -> execute -> validate -> cleaning
    until validation passes or the workflow is blocked.
    """

    csv_path = str(
        Path(csv_path).resolve()
    )

    if not Path(csv_path).exists():

        raise FileNotFoundError(
            f"Dataset not found: {csv_path}"
        )

    print("=" * 70)
    print("AUTONOMOUS DATA CLEANING WORKFLOW")
    print("=" * 70)
    print(f"Input dataset: {csv_path}")
    print()

    initial_state: AgentState = {
        "messages": [],
        "original_dataset_path": csv_path,
        "current_dataset_path": csv_path,
        "iteration": 0,
        "cleaning_history": [],
    }

    # LangGraph itself has a recursion safety setting.
    # This is NOT a cleaning retry limit.
    #
    # Increase it if you expect many correction cycles.
    result = graph.invoke(
        initial_state,
        config={
            "recursion_limit": 1000
        },
    )

    print("\n" + "=" * 70)
    print("WORKFLOW RESULT")
    print("=" * 70)

    print(
        result["messages"][-1].content
    )

    return result


# ============================================================
# 20. OPTIONAL GRAPH VISUALIZATION
# ============================================================

def print_graph():
    """
    Save the Mermaid representation of the graph as a PNG
    inside the save folder.
    """
    from pathlib import Path

    save_dir = Path("save")
    save_dir.mkdir(exist_ok=True)

    graph_image = graph.get_graph().draw_mermaid_png()

    output_path = save_dir / "workflow_graph.png"

    with open(output_path, "wb") as f:
        f.write(graph_image)

    print(f"[DONE] Graph saved to: {output_path}")


# ============================================================
# 21. MAIN
# ============================================================

if __name__ == "__main__":

    # Change this to your CSV file.
    CSV_PATH = "jsondata.csv"

    result = run_data_cleaning(
        CSV_PATH
    )

    # Uncomment if you want to see the graph:
    #
    print_graph()

    # Useful final values:
    #
    # print(result["understanding_report"])
    # print(result["validation_report"])
    # print(result["current_dataset_path"])
    # print(result["cleaning_history"])
