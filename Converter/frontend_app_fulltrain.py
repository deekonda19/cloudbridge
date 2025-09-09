from flask import Flask, render_template, request, jsonify
import subprocess
import os
from pathlib import Path

app = Flask(__name__)

# ----------------------
# Model listing helpers
# ----------------------
def list_hf_models():
    models = []
    ft_root = Path("finetuned_models")
    if ft_root.exists() and ft_root.is_dir():
        # include latest_ft if present
        latest = ft_root / "latest_ft"
        if latest.exists() and latest.is_dir():
            models.append(str(latest))
        # include all subdirs
        for p in sorted(ft_root.iterdir()):
            if p.is_dir() and p.name != "latest_ft":
                models.append(str(p))
    # include a couple of base models as fallbacks
    models.extend(["gpt2", "distilgpt2"])
    # de-dup while preserving order
    seen, uniq = set(), []
    for m in models:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq

'''def list_gpt4all_models():
    candidates = []
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
    found = sorted(set(found))
    return found'''
    
def list_gpt4all_models():
    candidates = []
    localapp = os.getenv("LOCALAPPDATA")
    if localapp:
        candidates.append(Path(localapp) / "nomic.ai" / "GPT4All")
    home = Path.home()
    candidates += [
        home / ".nomic" / "gpt4all",
        home / ".cache" / "gpt4all",
        home / "AppData" / "Local" / "nomic.ai" / "GPT4All",
    ]
    # Add fine-tuned latest models folder
    candidates.append(Path("finetuned_models") / "latest_ft")

    found = []
    for d in candidates:
        if d.exists() and d.is_dir():
            for pattern in ("*.gguf", "*.bin", "*.ggml"):
                found += [str(p) for p in d.glob(pattern)]
    found = sorted(set(found))
    return found


def list_ollama_models():
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        data = r.json()
        models = []
        if isinstance(data, dict) and "models" in data:
            for m in data["models"]:
                name = (m.get("name") or m.get("model") or m.get("tag"))
                if name:
                    models.append(name)
        elif isinstance(data, list):
            for m in data:
                if isinstance(m, dict):
                    name = (m.get("name") or m.get("model") or m.get("tag"))
                    if name:
                        models.append(name)
        return list(dict.fromkeys(models))
    except Exception:
        return []

# ----------------------
# Routes
# ----------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/models")
def models():
    backend = request.args.get("backend", "").strip().lower()
    if backend == "hf":
        return jsonify({"models": list_hf_models()})
    elif backend == "gpt4all":
        return jsonify({"models": list_gpt4all_models()})
    elif backend == "ollama":
        return jsonify({"models": list_ollama_models()})
    else:
        return jsonify({"models": []})

@app.route("/convert", methods=["POST"])
def convert():
    data = request.json
    folder = data.get("folder")
    backend = data.get("backend")
    model = data.get("model")
    use_latest = data.get("use_latest", False)
    no_auto_train = data.get("no_auto_train", False)

    if not folder or not backend or not model:
        return jsonify({"status": "error", "message": "Missing required inputs."}), 400

    cmd = [
        "python",
        "smart_pipeline_autotrain_latest_final_fulltrain.py",
        "--input", folder,
        "--backend", backend,
        "--model", model,
    ]
    if use_latest:
        cmd.append("--use_latest")
    if no_auto_train:
        cmd.append("--no_auto_train")

    try:
        subprocess.run(cmd, check=True)
        return jsonify({"status": "success", "message": "Conversion completed."})
    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    # prevent double execution in debug mode
    app.run(debug=True, use_reloader=False)
