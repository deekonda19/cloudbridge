# 🚀 Pentaho-to-PySpark Converter

This project provides a **conversion pipeline** and **Flask-based frontend** to automatically convert **Pentaho KTR/KJB files** into **PySpark code**, with optional **fine-tuning** using Hugging Face, GPT4All, or Ollama models.

---

## 📋 Features

- Convert **Pentaho `.ktr` and `.kjb` files** into PySpark scripts.  
- Supports **multiple backends**:
  - 🤗 Hugging Face (`transformers`)
  - 🦙 Ollama (local LLMs via `ollama serve`)
  - 🐧 GPT4All (local models in `.gguf`, `.bin`, `.ggml`)  
- Automatic dataset creation (`dataset_master.jsonl`) for fine-tuning.  
- **Auto-training** (Hugging Face backend only): fine-tune and save models to `finetuned_models/`.  
- Flask-based **frontend** (`frontend_app.py`) to run conversions via a simple web UI.  

---

## ⚙️ Prerequisites

1. **Python**: `>=3.9`  
2. **Pip packages** (install from `requirements.txt` you create):

   ```bash
   flask
   requests
   transformers
   torch
   gpt4all
   ```

   *(You may need `pip install torch --index-url https://download.pytorch.org/whl/cpu` if CUDA is not available.)*

3. **Optional backends**:
   - **Hugging Face**: Internet access required for model download (e.g. `gpt2`, `distilgpt2`, `starcoderbase`).
   - **GPT4All**: Place `.gguf`/`.bin`/`.ggml` models in one of these locations:
     - `C:\Users\<User>\AppData\Local\nomic.ai\GPT4All`
     - `~/.nomic/gpt4all`
     - `~/.cache/gpt4all`
   - **Ollama**: Install [Ollama](https://ollama.ai) and run:
     ```bash
     ollama serve
     ollama pull llama3:8b
     ```

---

## 📂 Folder Structure

```
project_root/
│
├── frontend_app.py                       # Flask frontend
├── smart_pipeline_autotrain_latest_final_f4.py   # Core pipeline
│
├── converted/                            # Output PySpark files (auto-created)
├── dataset_master.jsonl                   # Dataset of conversions (auto-updated)
├── finetuned_models/                      # Saved fine-tuned HF models
│   └── latest_ft/                        # Latest fine-tuned checkpoint
│
├── templates/
│   └── index.html                        # UI page for frontend_app
```

---

## ▶️ Usage

### 1. **Command Line (recommended for batch processing)**

Convert a Pentaho job or transformation file:

```bash
python smart_pipeline_autotrain_latest_final_f4.py --input ./pentaho_jobs --backend hf --model gpt2
```

Options:
- `--input <file_or_folder>` → Path to `.ktr`/`.kjb` file or folder  
- `--backend <hf|gpt4all|ollama>` → Backend to use  
- `--model <model_name_or_path>` → Model to use  
- `--use_latest` → Use the latest fine-tuned Hugging Face model  
- `--no_auto_train` → Skip auto-training after conversion  

#### Example Runs

1. **Hugging Face (with training):**
   ```bash
   python smart_pipeline_autotrain_latest_final_f4.py --input ./pentaho_jobs --backend hf --model gpt2
   ```

2. **Hugging Face (skip training):**
   ```bash
   python smart_pipeline_autotrain_latest_final_f4.py --input ./pentaho_jobs --backend hf --model gpt2 --no_auto_train
   ```

3. **GPT4All (local model):**
   ```bash
   python smart_pipeline_autotrain_latest_final_f4.py --input ./pentaho_jobs --backend gpt4all --model C:\Users\venki\AppData\Local\nomic.ai\GPT4All\mymodel.gguf
   ```

4. **Ollama:**
   ```bash
   python smart_pipeline_autotrain_latest_final_f4.py --input ./pentaho_jobs --backend ollama --model llama3:8b
   ```

---

### 2. **Flask Frontend (Web UI)**

Run the Flask app:

```bash
python frontend_app.py
```

Access in browser: [http://localhost:5000](http://localhost:5000)

Features:
- Select backend (`HF`, `GPT4All`, `Ollama`)  
- Auto-list available models  
- Choose options (`use_latest`, `skip auto-train`)  
- Convert folder with one click  

---

## 📤 Output

- Converted PySpark files → stored in `converted/<model_timestamp>/`  
  Example:
  ```
  converted/gpt2_20250824_153200/job1_converted.py
  converted/gpt2_20250824_153200/transformation1_converted.py
  ```

- Dataset → appended to `dataset_master.jsonl` for continuous fine-tuning.  

- Fine-tuned models → saved in `finetuned_models/ft_<model_name>_<timestamp>/`  
  - Also copied to `finetuned_models/latest_ft/`  

---

## 🧪 Sample Expected Output

Example conversion from a simple Pentaho step:

**Pentaho step**:
```xml
<step>
  <name>FilterRows</name>
  <type>FilterRows</type>
</step>
```

**Generated PySpark**:
```python
from pyspark.sql.functions import col

df_filtered = df.filter(col("some_column") > 0)
```

---
