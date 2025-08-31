# 📘 Cloud Bridge Validator --- Setup & Run Guide

This Streamlit app validates, corrects, and analyzes PySpark code
converted from Pentaho ETL XMLs (`.ktr` / `.kjb`).

------------------------------------------------------------------------

## 1. ✅ Prerequisites

-   **Python 3.9+** (tested on 3.9--3.12)\
-   **Pip** (Python package manager)\
-   OS: Windows, macOS, or Linux

### Required Python Libraries

-   `streamlit`
-   `pandas`
-   `openpyxl` (for Excel export)
-   `gpt4all` (if you want GPT4All local models)
-   `ollama` (if you want Ollama local models, needs Ollama installed
    separately)

------------------------------------------------------------------------

## 2. 📦 Installation of Prerequisites

### Step 1: Install Python

-   **Windows**: Download from
    [python.org](https://www.python.org/downloads/) and check *"Add to
    PATH"* during installation.\
-   **Linux/macOS**: Already available, or install via package manager.

Check installation:

``` bash
python --version
pip --version
```

------------------------------------------------------------------------

### Step 2: Install Required Python Libraries

Run:

``` bash
pip install streamlit pandas openpyxl gpt4all
```

⚠️ If you will **use Ollama** for LLM-based corrections: - Install
Ollama: <https://ollama.com/download>\
- Then pull a model (e.g. `mistral`):

``` bash
ollama pull mistral
```

⚠️ If you will **use GPT4All**: - Download a `.bin` or `.gguf` model
file from <https://gpt4all.io/index.html>\
- Place it in one of these folders: - `~/.gpt4all` (Linux/macOS)\
- `C:\Users\<YOU>\AppData\Local\nomic.ai\GPT4All` (Windows)\
- `./models` (relative to the script)\
- `~/.noai`

------------------------------------------------------------------------

## 3. 📂 Recommended Folder Structure

    project-root/
    ├── pyspvalidator_stlit_f7.py     # main app
    ├── output_pyspark/               # corrected/generated PySpark code (auto-created)
    ├── models/                       # optional GPT4All local models (.bin / .gguf)
    ├── pentaho_jobs/                 # input Pentaho .ktr/.kjb files
    ├── pyspark_converted/            # input converted PySpark .py files (if available)
    └── README.md

------------------------------------------------------------------------

## 4. ▶️ Running the App

Navigate to the folder containing `pyspvalidator_stlit_f7.py`, then run:

``` bash
streamlit run pyspvalidator_stlit_f7.py
```

This will open a **local web app** in your browser (default at
<http://localhost:8501>).

------------------------------------------------------------------------

## 5. ⚙️ Usage

1.  **Select Workflow Mode**
    -   *Complexity Analysis (XML only)* → Just analyze ETL complexity.\
    -   *Compare & Correct* → Fix existing PySpark code (keeps or
        overwrites).
2.  **Set Input Folders**
    -   XML folder: where your `.ktr` / `.kjb` files live.\
    -   PySpark folder: where converted `.py` files live (if any).
3.  **Set Output Folder**
    -   Defaults to `output_pyspark` (created if missing).
4.  **Choose Engine**
    -   *Rule-based (no LLM)* → Simple string/rule corrections.\
    -   *LLM (Ollama)* → Uses a local Ollama model.\
    -   *LLM (GPT4All)* → Uses a local GPT4All model.
5.  **Click Run**
    -   Generates corrected/validated code.\
    -   Saves Excel report (with timestamped filename) summarizing
        results.

------------------------------------------------------------------------

## 6. 📊 Output

-   **Corrected / Generated PySpark files** → in `output_pyspark/`\
-   **Excel Report** → `correction_report_<timestamp>.xlsx` or
    `complexity_report_<timestamp>.xlsx`
    -   Includes complexity stats, accuracy before/after, and whether
        file was *corrected* or *generated*.
