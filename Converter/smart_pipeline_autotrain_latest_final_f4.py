import os
import re
import argparse
import json
import shutil
import requests
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# -------------------
# Utility Functions
# -------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def sanitize_filename(name: str) -> str:
    """Sanitize folder/file names for Windows."""
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

def clean_generated_code(text: str) -> str:
    """Extract valid PySpark code from model output, drop guidelines and explanations."""
    lines = text.splitlines()
    code_lines = []
    inside_code = False
    for line in lines:
        # Detect fenced code blocks
        if line.strip().startswith("```"):
            inside_code = not inside_code
            continue

        # If inside code block, keep everything
        if inside_code:
            code_lines.append(line)
            continue

        # Otherwise, keep only lines that look like Python/PySpark
        if re.match(r"^\s*(from |import |df|spark|withColumn|filter|join|select|col\()", line):
            code_lines.append(line)

    return "\n".join(code_lines).strip()

# -------------------
# Pentaho Parsing
# -------------------

def parse_pentaho_file(file_path: str):
    steps = []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        for step in root.findall(".//step"):
            name = step.find("name").text if step.find("name") is not None else "UnknownStep"
            stype = step.find("type").text if step.find("type") is not None else "UnknownType"
            steps.append({"name": name, "type": stype})
    except Exception as e:
        print(f"[ERROR] Failed to parse {file_path}: {e}")
    return steps

# -------------------
# Code Generation
# -------------------

def generate_pyspark_code(backend: str, model: str, steps):
    if backend == "hf":
        generator = pipeline("text-generation", model=model)
        prompt = f"Convert the following Pentaho steps into equivalent PySpark code:\n{steps}\n"
        raw_output = generator(prompt, max_length=512, do_sample=True)[0]["generated_text"]
        return clean_generated_code(raw_output)

    elif backend == "ollama":
        prompt = f"Convert the following Pentaho steps into equivalent PySpark code:\n{steps}\n"
        r = requests.post("http://localhost:11434/api/generate", json={"model": model, "prompt": prompt, "stream": False})
        if r.status_code == 200:
            raw_output = r.json().get("response", "")
            return clean_generated_code(raw_output)
        else:
            print(f"[ERROR] Ollama request failed: {r.text}")
            return ""

    elif backend == "gpt4all":
        from gpt4all import GPT4All
        gpt = GPT4All(model)
        #gpt.open()
        prompt = f"Convert the following Pentaho steps into equivalent PySpark code:\n{steps}\n"
        #raw_output = gpt.prompt(prompt)
        with gpt.chat_session():
            raw_output = gpt.generate(prompt, max_tokens=512)
        return clean_generated_code(raw_output)

    else:
        raise ValueError(f"Unsupported backend: {backend}")

# -------------------
# Dataset Handling
# -------------------

DATASET_PATH = Path("dataset_master.jsonl")

def append_to_dataset(prompt: str, completion: str):
    if not prompt.strip() or not completion.strip():
        return
    entry = {
        "prompt": prompt.lower().strip(),
        "completion": completion.lower().strip()
    }
    with open(DATASET_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

# -------------------
# Training
# -------------------

def auto_train_latest(model: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("finetuned_models") / f"ft_{sanitize_filename(Path(model).name)}_{timestamp}"
    ensure_dir(output_dir)

    from transformers import Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

    tokenizer = AutoTokenizer.from_pretrained(model)
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=str(DATASET_PATH),
        block_size=128
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = AutoModelForCausalLM.from_pretrained(model)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir=str(output_dir / "logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()

    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Also copy as latest
    latest_dir = Path("finetuned_models") / "latest_ft"
    if latest_dir.exists():
        if latest_dir.is_dir():
            shutil.rmtree(latest_dir)
        else:
            latest_dir.unlink()
    shutil.copytree(output_dir, latest_dir)

    print(f"[INFO] Fine-tuned model saved at {output_dir}, latest model updated.")

# -------------------
# Auto-list helpers (added)
# -------------------

def list_hf_models(use_latest: bool):
    models = []
    ft_root = Path("finetuned_models")
    if use_latest:
        latest = ft_root / "latest_ft"
        if latest.exists():
            return ["latest_ft"]
    if ft_root.exists() and ft_root.is_dir():
        for p in sorted(ft_root.iterdir()):
            if p.is_dir():
                models.append(p.name)
    # Add a few base models for convenience
    models.extend(["gpt2", "distilgpt2","Salesforce/codegen-350M-mono","bigcode/starcoderbase"])
    # De-dup while preserving order
    seen = set()
    uniq = []
    for m in models:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq

def list_gpt4all_models():
    candidates = []
    # Common Windows locations
    localapp = os.getenv("LOCALAPPDATA")
    if localapp:
        candidates.append(Path(localapp) / "nomic.ai" / "GPT4All")
    home = Path.home()
    candidates += [
        home / ".nomic" / "gpt4all",
        home / ".cache" / "gpt4all",
        home / "AppData" / "Local" / "nomic.ai" / "GPT4All",
    ]
    found = []
    for d in candidates:
        if d.exists() and d.is_dir():
            for pattern in ("*.gguf", "*.bin", "*.ggml"):
                found += [str(p) for p in d.glob(pattern)]
    # Present just file names to user, but we will pass the same string back
    if not found:
        return []
    # Sort stable
    found = sorted(found)
    return found

def list_ollama_models():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        data = r.json()
        models = []
        if isinstance(data, dict) and "models" in data:
            for m in data["models"]:
                # try various keys defensively
                name = m.get("name") or m.get("model") or m.get("tag")
                if name:
                    models.append(name)
        elif isinstance(data, list):
            for m in data:
                name = (m.get("name") if isinstance(m, dict) else None)
                if name:
                    models.append(name)
        # de-dup
        models = list(dict.fromkeys(models))
        return models
    except Exception:
        return []

# -------------------
# Main
# -------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to a Pentaho .ktr/.kjb file or folder")
    parser.add_argument("--backend", help="Backend to use: hf, gpt4all, ollama")
    parser.add_argument("--model", help="Model name/path")
    parser.add_argument("--use_latest", action="store_true", help="Use latest fine-tuned HF model")
    parser.add_argument("--no_auto_train", action="store_true", help="Skip auto-training")
    args = parser.parse_args()

    input_path = Path(args.input)

    # Detect backend/model interactively if not given
    backend = args.backend
    model = args.model

    if not backend or not model:
        print("Choose backend:")
        print("1. Hugging Face")
        print("2. GPT4All")
        print("3. Ollama")
        sel = input("Enter choice: ").strip()
        if sel == "1":
            backend = "hf"
            # auto-list models
            models = list_hf_models(use_latest=args.use_latest)
            if not models:
                print("[WARN] No local fine-tuned HF models found; you can still enter a base model like 'gpt2'.")
                models = ["gpt2", "distilgpt2"]
            print("\nAvailable Hugging Face models:")
            for i, m in enumerate(models, 1):
                print(f"{i}. {m}")
            print("0. Enter manually")
            ch = input("Choose model: ").strip()
            if ch == "0":
                model = input("Enter Hugging Face model (e.g. gpt2 or path): ").strip()
            else:
                idx = int(ch) - 1
                chosen = models[idx]
                # resolve latest_ft to full path
                if chosen == "latest_ft":
                    model = str(Path("finetuned_models") / "latest_ft")
                elif (Path("finetuned_models") / chosen).exists():
                    model = str(Path("finetuned_models") / chosen)
                else:
                    model = chosen

        elif sel == "2":
            backend = "gpt4all"
            models = list_gpt4all_models()
            if models:
                print("\nDetected GPT4All models:")
                for i, m in enumerate(models, 1):
                    print(f"{i}. {m}")
                print("0. Enter manually")
                ch = input("Choose model: ").strip()
                if ch == "0":
                    model = input("Enter GPT4All model name or path: ").strip()
                else:
                    model = models[int(ch) - 1]
            else:
                print("[WARN] No local GPT4All models detected; you can still enter a known model name.")
                model = input("Enter GPT4All model name: ").strip()

        elif sel == "3":
            backend = "ollama"
            models = list_ollama_models()
            if models:
                print("\nDetected Ollama models:")
                for i, m in enumerate(models, 1):
                    print(f"{i}. {m}")
                print("0. Enter manually")
                ch = input("Choose model: ").strip()
                if ch == "0":
                    model = input("Enter Ollama model name (e.g. llama3:8b): ").strip()
                else:
                    model = models[int(ch) - 1]
            else:
                print("[WARN] Could not list Ollama models; ensure Ollama is running. You can still enter a model name.")
                model = input("Enter Ollama model name (e.g. llama3:8b): ").strip()
        else:
            raise ValueError("Invalid choice")

    print(f"[INFO] Running conversion with backend: {backend}, model: {model}")

    # Output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = sanitize_filename(model)
    outdir = Path("converted") / f"{model_safe}_{timestamp}"
    ensure_dir(outdir)

    files = [input_path] if input_path.is_file() else list(input_path.glob("*.ktr")) + list(input_path.glob("*.kjb"))

    for f in files:
        print(f"[INFO] Converting {f}")
        steps = parse_pentaho_file(str(f))
        pyspark_code = generate_pyspark_code(backend, model, steps)

        # Save converted code
        outfile = outdir / f"{f.stem}_converted.py"
        with open(outfile, "w", encoding="utf-8") as out:
            out.write(pyspark_code)

        # Append to dataset
        append_to_dataset(str(steps), pyspark_code)

    # Auto-train (HF only)
    if backend == "hf" and not args.no_auto_train:
        auto_train_latest(model)

if __name__ == "__main__":
    main()
