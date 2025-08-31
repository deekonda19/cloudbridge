import os, re, json, subprocess, difflib
import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher

st.set_page_config(page_title="Cloud Bridge Validator", layout="wide")
st.title("Cloud Bridge Validator")

# ===================== HELPERS =====================

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def write_text(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def normalize_stem(name: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9]", "_", name).lower()
    base = re.sub(r"(_converted|_corrected)$", "", base)
    return base

def list_map_by_stem(folder: str, exts: set):
    files = {}
    for fn in os.listdir(folder):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in exts:
            files[normalize_stem(stem)] = os.path.join(folder, fn)
    return files

def diff_rows(orig: str, new: str, file_label: str):
    rows = []
    diff = list(difflib.unified_diff(
        orig.splitlines(), new.splitlines(),
        fromfile=f"{file_label} (original)",
        tofile=f"{file_label} (new)",
        lineterm=""
    ))
    for line in diff:
        if line.startswith("@@"):
            rows.append({"rule": "HUNK", "before": "", "after": line, "note": ""})
        elif line.startswith("+") and not line.startswith("+++"):
            rows.append({"rule": "DIFF_INSERT", "before": "", "after": line[1:], "note": ""})
        elif line.startswith("-") and not line.startswith("---"):
            rows.append({"rule": "DIFF_DELETE", "before": line[1:], "after": "", "note": ""})
    return rows

# ------------------ Accuracy helper ------------------
def compute_accuracy(code_in: str, code_out: str, expected: dict) -> tuple:
    total = 0
    score_in = 0
    score_out = 0

    # Joins
    if expected.get("joins"):
        total += 1
        if ".join(" in code_in: score_in += 1
        if ".join(" in code_out: score_out += 1

    # Filters
    if expected.get("filters"):
        total += 1
        if ".filter(" in code_in: score_in += 1
        if ".filter(" in code_out: score_out += 1

    # Columns
    if expected.get("columns"):
        total += 1
        if any(x in code_in for x in [".select(", ".withColumn", ".withColumnRenamed"]): score_in += 1
        if any(x in code_out for x in [".select(", ".withColumn", ".withColumnRenamed"]): score_out += 1

    # String ops
    if expected.get("string_ops"):
        total += 1
        if ".withColumn" in code_in and any(fn in code_in for fn in ["upper(", "lower(", "substr("]): score_in += 1
        if ".withColumn" in code_out and any(fn in code_out for fn in ["upper(", "lower(", "substr("]): score_out += 1

    # Calculations
    if expected.get("calculations"):
        total += 1
        if ".withColumn" in code_in and any(op in code_in for op in ["+", "-", "*", "/"]): score_in += 1
        if ".withColumn" in code_out and any(op in code_out for op in ["+", "-", "*", "/"]): score_out += 1

    # Output
    if expected.get("output") or "write" in code_in or "write" in code_out:
        total += 1
        if ".write" in code_in: score_in += 1
        if ".write" in code_out: score_out += 1

    if total == 0:
        return (50.0, 50.0)

    return (score_in / total * 100.0, score_out / total * 100.0)

# ------------------ XML Expectations Parser ------------------
def basic_xml_expectations(xml_text: str) -> dict:
    expected = {
        "columns": [],
        "filters": [],
        "joins": [],
        "calculations": [],
        "string_ops": []
    }
    try:
        root = ET.fromstring(xml_text)
        for field in root.findall(".//field"):
            name = field.findtext("name")
            if name:
                expected["columns"].append(name.strip())
        for name in root.findall(".//name"):
            if name.text and len(name.text.strip()) > 0:
                expected["columns"].append(name.text.strip())
        for cond in root.findall(".//condition"):
            txt = "".join(cond.itertext()).strip()
            if txt:
                expected["filters"].append(txt)
        for compare in root.findall(".//compare"):
            txt = "".join(compare.itertext()).strip()
            if txt:
                expected["filters"].append(txt)
        for join in root.findall(".//MergeJoinMeta"):
            jtype = join.findtext("join_type")
            if jtype:
                expected["joins"].append(jtype.strip())
        for step in root.findall(".//step"):
            stype = step.findtext("type")
            if stype and "Join" in stype:
                expected["joins"].append(stype.strip())
        for calc in root.findall(".//CalculatorMetaFunction"):
            fieldname = calc.findtext("field_name")
            calctype = calc.findtext("calc_type")
            if fieldname and calctype:
                expected["calculations"].append(f"{fieldname}:{calctype}")
        for calc in root.findall(".//calculation"):
            txt = "".join(calc.itertext()).strip()
            if txt:
                expected["calculations"].append(txt)
        string_ops_keywords = ["Trim", "ReplaceString", "LowerCase", "UpperCase", "Substring"]
        for step in root.findall(".//step"):
            stype = step.findtext("type")
            if stype and any(k in stype for k in string_ops_keywords):
                expected["string_ops"].append(stype.strip())
        for k in expected:
            expected[k] = list(set(expected[k]))
    except Exception as e:
        print(f"XML parse error: {e}")
    return expected

# ------------------ Complexity Analyzer ------------------
def analyze_complexity(xml_text: str) -> dict:
    expected = basic_xml_expectations(xml_text)
    try:
        root = ET.fromstring(xml_text)
        hops_count = len(root.findall(".//step"))  # each transformation node
    except Exception:
        hops_count = 0

    stats = {
        "columns_count": len(expected["columns"]),
        "filters_count": len(expected["filters"]),
        "joins_count": len(expected["joins"]),
        "calculations_count": len(expected["calculations"]),
        "string_ops_count": len(expected["string_ops"]),
        "xml_lines": len(xml_text.splitlines()),
        "hops_count": hops_count
    }

    score = (
        stats["columns_count"] * 1 +
        stats["filters_count"] * 2 +
        stats["joins_count"] * 3 +
        stats["calculations_count"] * 3 +
        stats["string_ops_count"] * 2
    )
    if score < 10:
        level = "Low"
    elif score < 25:
        level = "Medium"
    else:
        level = "High"

    stats["complexity_level"] = level
    stats["total_score"] = score
    return stats

# ------------------ Ollama helpers ------------------
def run_ollama(model: str, prompt: str) -> str:
    proc = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout.decode("utf-8", errors="ignore")

def list_ollama_models() -> list:
    try:
        proc = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if proc.returncode != 0:
            return []
        lines = proc.stdout.splitlines()
        return [line.split()[0] for line in lines[1:] if line.strip()]
    except Exception:
        return []

# ------------------ GPT4All helpers ------------------
def list_gpt4all_models() -> list:
    models = []
    home = os.path.expanduser("~/.gpt4all")
    local = os.path.join(os.getcwd(), "models")
    noai = os.path.expanduser("~/.noai")
    win_appdata = os.path.join(os.path.expanduser("~"), "AppData", "Local", "nomic.ai", "GPT4All")

    for path in [home, local, noai, win_appdata]:
        if os.path.isdir(path):
            for fn in os.listdir(path):
                if fn.endswith(".bin") or fn.endswith(".gguf"):
                    models.append(os.path.join(path, fn))
    return models


def run_gpt4all(model: str, prompt: str) -> str:
    try:
        from gpt4all import GPT4All
        gptj = GPT4All(model)
        with gptj.chat_session():
            return gptj.generate(prompt)
    except Exception as e:
        return f"# GPT4All run not available: {e}"

# ------------------ Simplified rules ------------------
def rule_correct(code: str, expected: dict):
    fixes = []
    corrected = code

    # Fix triple equals
    if "===" in corrected:
        corrected = corrected.replace("===", "==")
        fixes.append({"rule": "Replace ===", "note": "PySpark uses '==' not '==='."})

    # Fix malformed join
    join_pattern = re.compile(r"\.with\s*==.*", re.MULTILINE)
    if join_pattern.search(corrected):
        corrected = join_pattern.sub(
            "# FIXED: replaced malformed join → specify join condition manually", corrected
        )
        fixes.append({"rule": "Fix malformed join", "note": "Replaced broken join with placeholder"})

    # Remove invalid backslash+comment
    corrected_new = re.sub(r"\\\s*#.*", r"\\", corrected)
    if corrected_new != corrected:
        fixes.append({"rule": "Clean line continuation", "note": "Removed invalid comment after backslash"})
        corrected = corrected_new

    # Fix .mode line continuation
    corrected_new = corrected.replace('.mode("overwrite") \\', '.mode("overwrite")\\')
    if corrected_new != corrected:
        fixes.append({"rule": "Fix mode continuation", "note": "Fixed line continuation"})
        corrected = corrected_new

    if corrected == code:
        fixes.append({"rule": "NO_CHANGE", "note": "No rule-based changes applied"})

    return corrected, fixes

def llm_correct(code: str, expected: dict, engine_key: str, model: str = None) -> str:
    return f"# LLM corrected PySpark\n{code}"

# ------------------ NEW: Generate PySpark from XML ------------------
def generate_pyspark_from_xml(xml_text: str, expected: dict, engine_choice: str, model=None) -> str:
    if engine_choice == "LLM (Ollama)" and model:
        return run_ollama(model, f"Convert this ETL XML into equivalent PySpark code:\n{xml_text}")
    elif engine_choice == "LLM (GPT4All)" and model:
        return run_gpt4all(model, f"Convert this ETL XML into equivalent PySpark code:\n{xml_text}")
    else:
        # Fallback
        return f"# Could not generate automatically.\n# Expected elements: {json.dumps(expected, indent=2)}\n"

# ===================== UI =====================

mode = st.radio("Workflow Mode", [
    "Complexity Analysis (XML only)",
    "Compare & Correct (keep original + corrected)",
    "Compare & Correct (overwrite original)"
], horizontal=False)

col1, col2 = st.columns(2)
with col1:
    xml_dir = st.text_input("📁 XML folder (.ktr / .kjb)", value="")
with col2:
    py_dir  = st.text_input("📁 Converted PySpark folder (.py)", value="")

out_dir = st.text_input("📁 Output folder", value="output_pyspark")

engine_choice = st.radio("", [
    "Rule-based (no LLM)",
    "LLM (Ollama)",
    "LLM (GPT4All)"
], index=0, horizontal=True)

ollama_model = None
gpt4all_model = None

if engine_choice == "LLM (Ollama)":
    models = list_ollama_models()
    if models:
        ollama_model = st.selectbox("Ollama model", models, index=0)
    else:
        st.warning("⚠️ No Ollama models detected. Try `ollama pull mistral` or similar.")

elif engine_choice == "LLM (GPT4All)":
    models = list_gpt4all_models()
    if models:
        gpt4all_model = st.selectbox("GPT4All model", models, index=0)
    else:
        st.warning("⚠️ No GPT4All models detected. Place `.bin` models in ~/.gpt4all or ./models")

# ===================== PROCESS =====================

if st.button("Run", type="primary"):
    if not xml_dir or not os.path.isdir(xml_dir):
        st.error("Please provide a valid XML folder.")
    else:
        os.makedirs(out_dir, exist_ok=True)
        xml_map = list_map_by_stem(xml_dir, {".ktr", ".kjb"})
        py_map  = list_map_by_stem(py_dir,  {".py"}) if py_dir and os.path.isdir(py_dir) else {}

        all_rows = []
        processed = 0

        for stem, xml_path in xml_map.items():
            processed += 1
            xml_text = read_text(xml_path)
            expected = basic_xml_expectations(xml_text)
            had_existing = stem in py_map
            code_in = read_text(py_map[stem]) if had_existing else ""
            code_out = code_in

            # --- complexity only ---
            if mode == "Complexity Analysis (XML only)":
                stats = analyze_complexity(xml_text)
                stats["file"] = stem
                all_rows.append(stats)
                st.write(
                    f"📊 {stem}: {stats['complexity_level']} complexity "
                    f"(XML lines={stats['xml_lines']}, Hops={stats['hops_count']}, "
                    f"Cols={stats['columns_count']}, Filters={stats['filters_count']}, "
                    f"Joins={stats['joins_count']}, Calcs={stats['calculations_count']}, "
                    f"StringOps={stats['string_ops_count']})"
                )
                continue

            # --- correction / generation ---
            if had_existing:
                if engine_choice == "Rule-based (no LLM)":
                    code_out, _ = rule_correct(code_in, expected)
                elif engine_choice == "LLM (Ollama)":
                    code_out = llm_correct(code_in, expected, "ollama", ollama_model)
                elif engine_choice == "LLM (GPT4All)":
                    code_out = run_gpt4all(gpt4all_model, f"Correct PySpark for: {json.dumps(expected)}\n\nCode:\n{code_in}")
            else:
                # No existing PySpark → generate fresh
                st.warning(f"⚠️ No matching PySpark file found for {stem}. Generating new code.")
                code_out = generate_pyspark_from_xml(xml_text, expected, engine_choice,
                                                     ollama_model if engine_choice=="LLM (Ollama)" else gpt4all_model)

            # --- Accuracy scoring ---
            acc_before, acc_after = compute_accuracy(code_in, code_out, expected)
            st.write(f"📊 Accuracy before: {acc_before:.1f}% | after: {acc_after:.1f}%")

            # If accuracy too low, regenerate from XML
            if acc_after < 20.0:
                st.warning(f"⚠️ Accuracy very low for {stem}. Regenerating from XML.")
                code_out = generate_pyspark_from_xml(xml_text, expected, engine_choice,
                                                     ollama_model if engine_choice=="LLM (Ollama)" else gpt4all_model)

            # --- save filename logic ---
            if mode == "Compare & Correct (keep original + corrected)" and had_existing:
                out_file = os.path.join(out_dir, f"{stem}_corrected.py")
                st.write(f"🔧 Corrected → {out_file}")
                action = "corrected"
            elif mode == "Compare & Correct (overwrite original)" and had_existing:
                out_file = py_map[stem]
                st.write(f"✏️ Overwrote → {out_file}")
                action = "corrected"
            else:
                out_file = os.path.join(out_dir, f"{stem}_generated.py")
                st.write(f"🆕 Generated → {out_file}")
                action = "generated"

            write_text(out_file, code_out)

            # --- log diffs ---
            dr = diff_rows(code_in, code_out, stem) if had_existing else []
            for r in dr:
                r.update({
                    "file": stem,
                    "mode": mode,
                    "engine": engine_choice,
                    "accuracy_before": acc_before,
                    "accuracy_after": acc_after,
                    "action": action
                })
            if not dr:  # for generated cases with no diff
                all_rows.append({
                    "file": stem,
                    "mode": mode,
                    "engine": engine_choice,
                    "accuracy_before": acc_before,
                    "accuracy_after": acc_after,
                    "action": action
                })

        # --- Excel log + summary ---
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if mode == "Complexity Analysis (XML only)":
            prefix = "complexity_report"
        else:
            prefix = "correction_report"

        log_path = os.path.join(out_dir, f"{prefix}_{timestamp}.xlsx")
        if all_rows:
            df = pd.DataFrame(all_rows)
            df.to_excel(log_path, index=False)
            st.success(f"✅ Processed {processed} XML file(s). Report: {log_path}")
            st.dataframe(df)

            if mode != "Complexity Analysis (XML only)" and \
               "accuracy_before" in df.columns and "accuracy_after" in df.columns:
                avg_before = df["accuracy_before"].mean()
                avg_after = df["accuracy_after"].mean()
                st.markdown("### 📊 Summary Accuracy (all files)")
                st.write(f"**Before correction:** {avg_before:.1f}%")
                st.write(f"**After correction:** {avg_after:.1f}%")
                st.write(f"**Improvement:** {avg_after - avg_before:+.1f}%")
        else:
            st.info(f"Processed {processed} XML file(s). No diffs recorded.")
