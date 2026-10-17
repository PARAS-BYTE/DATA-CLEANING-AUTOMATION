# AI CSV Cleaner

Google Sheet where the transformations were executed, including the cleaned dataset:

[Open the Google Sheet](https://docs.google.com/spreadsheets/d/1_EHw9z5mFbCwHYfrgdyRTws3X4a41F4D62qVjT0z8xo/edit?usp=sharing)

## Overview

A simple Streamlit app where AI agents inspect, clean and validate your CSV file, and you can watch every step live.

## What it does

1. You upload a CSV file.
2. The app shows a quick overview: rows, columns, missing values, duplicates, column info and numeric summary.
3. You click **Clean dataset**.
4. Four workers go through the data one after another:

| Step | Worker | Job |
|------|--------|-----|
| 1 | **Understanding Agent** | Reads the data profile, decides if cleaning is needed and lists every issue |
| 2 | **Cleaning Agent** | Writes a pandas script that fixes only those issues |
| 3 | **Execution Engine** | Runs the script in a separate process (not an AI) and compares before/after |
| 4 | **Validation Agent** | Checks that issues are fixed and that no valid data was lost |

5. If validation fails, the Cleaning Agent tries again with the feedback (up to the max attempts you choose).
6. The cleaned data is shown directly in the app with before/after numbers. You can download it if you want.

```
Understand -> Clean -> Execute -> Validate -> Finish
                ^         |          |
                +---------+----------+  (retry if something fails)
```

## Setup

```bash
pip install -U streamlit pandas numpy langchain-core langchain-groq langgraph
streamlit run app.py
```

Then open the app, paste your **Groq API key** in the sidebar (get one at [console.groq.com](https://console.groq.com)), and upload a CSV.

## Good to know

- **The API key is never stored in the code.** You enter it in the sidebar each time.
- **No files are saved in your project folder.** Working files go to a temporary folder that is deleted after each run. The cleaned CSV is only saved if you click the download button.
- **Generated scripts run on your computer.** Use only data you trust.
- You can change the model and the maximum number of attempts in the sidebar.

## Project files

- `app.py`: the whole app (prompts, agents, workflow and Streamlit interface)
- `README.md`: this file