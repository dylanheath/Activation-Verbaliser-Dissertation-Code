#!/usr/bin/env python3
"""
run_nla_demo.py
================

End-to-end runner for the Natural Language Autoencoder (NLA) worked example.

What this script does, in order:
  1. Verifies environment (CUDA, free VRAM, required packages).
  2. Pre-downloads the three HuggingFace artifacts (base model, AV, AR).
  3. Extracts layer-20 residual-stream activations from Qwen2.5-7B-Instruct
     for a set of user-defined prompts, saved as parquet files.
  4. Launches the SGLang server hosting the AV (activation verbalizer)
     as a background subprocess.
  5. Calls nla_inference.py for each parquet, capturing verbalisations.
  6. Shuts down the AV server, loads the AR (activation reconstructor),
     computes MSE between original activations and AR-reconstructed
     vectors derived from the verbalisations.
  7. Writes a consolidated results JSON and a CSV table ready to drop
     into the dissertation chapter.
  8. Tarballs everything in ./results/ for easy scp off the pod.

Assumes:
  - You are running on a RunPod (or similar) Linux box with an A100-class
    GPU (>=40GB VRAM recommended).
  - You have cloned https://github.com/kitft/natural_language_autoencoders
    and this script lives at the repo root (alongside nla_inference.py).
  - Required deps installed:
        pip install torch transformers safetensors httpx orjson pyyaml \\
                    numpy pyarrow huggingface_hub "sglang[all]>=0.5.6"

Usage:
    python run_nla_demo.py              # full pipeline, default prompts
    python run_nla_demo.py --skip-download  # if you've already pulled weights
    python run_nla_demo.py --extract-only   # just step 3, no server work
    python run_nla_demo.py --prompts-file my_prompts.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration. Edit the PROMPTS dict to control what gets probed.
# ---------------------------------------------------------------------------

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
AV_MODEL   = "kitft/nla-qwen2.5-7b-L20-av"
AR_MODEL   = "kitft/nla-qwen2.5-7b-L20-ar"
LAYER      = 20            # NLA was trained on layer 20 of this base model.
D_MODEL    = 3584          # Qwen2.5-7B hidden size; used as a sanity check.
SGLANG_PORT = 30000

# Default prompts: a benign control set. To probe an adversarial example
# from the 17-session dataset, add it under a new key locally. The script
# extracts forward-pass activations only — no completions are generated
# from these prompts, so the model never produces harmful output.
DEFAULT_PROMPTS: dict[str, str] = {
    "benign_science":  "Explain how photosynthesis works in plants.",
    "benign_code":     "Write a Python function that returns the nth Fibonacci number.",
    "benign_neutral":  "The Thames flows through London and empties into the North Sea.",
    # Add e.g. "structural_probe": "<prompt from your dataset>" here, locally.
}

RESULTS_DIR = Path("results")
ACTS_DIR    = RESULTS_DIR / "activations"
VERB_DIR    = RESULTS_DIR / "verbalisations"
LOGS_DIR    = RESULTS_DIR / "logs"

# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def die(msg: str, code: int = 1) -> None:
    log(f"FATAL: {msg}")
    sys.exit(code)


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, ACTS_DIR, VERB_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_for_port(port: int, timeout: float = 600.0) -> bool:
    """Poll until SGLang's port accepts connections, or time out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(2)
    return False


# ---------------------------------------------------------------------------
# Step 1: environment check
# ---------------------------------------------------------------------------

def check_environment() -> None:
    log("Step 1/8: environment check")

    # Required Python packages
    missing = []
    for pkg in ("torch", "transformers", "pyarrow", "huggingface_hub",
                "safetensors", "yaml", "httpx", "orjson"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        die(f"missing packages: {missing}. Install with pip first.")

    # nla_inference.py is expected at the repo root next to this script
    if not Path("nla_inference.py").exists():
        die("nla_inference.py not found in the current directory. "
            "Run this script from the cloned natural_language_autoencoders repo root.")

    import torch
    if not torch.cuda.is_available():
        die("CUDA not available. This script requires a GPU.")

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    log(f"GPU: {props.name}, {vram_gb:.1f} GB VRAM")
    if vram_gb < 38:
        log("WARNING: <40GB VRAM detected. Qwen2.5-7B in bf16 plus SGLang "
            "KV cache may OOM. Consider quantisation or a larger card.")


# ---------------------------------------------------------------------------
# Step 2: pre-download weights
# ---------------------------------------------------------------------------

def download_weights() -> None:
    log("Step 2/8: pre-downloading weights (this is the long one)")
    from huggingface_hub import snapshot_download
    for repo_id in (BASE_MODEL, AV_MODEL, AR_MODEL):
        log(f"  pulling {repo_id}")
        snapshot_download(repo_id=repo_id, resume_download=True)
    log("  all weights cached")


# ---------------------------------------------------------------------------
# Step 3: extract activations
# ---------------------------------------------------------------------------

def extract_activations(prompts: dict[str, str],
                        max_tokens: int = 8192) -> dict[str, dict[str, Any]]:
    """Extract layer-20 hidden states from Qwen2.5-7B-Instruct.

    Long prompts (e.g. full multi-turn conversation transcripts) are
    truncated to ``max_tokens`` before the forward pass to keep VRAM use
    bounded. Truncation keeps the *start* of the conversation, which
    preserves the structural framing — for analytical work where the
    setup is what's interesting, this is usually what you want. If you
    care about late-turn activations specifically, take the tail instead
    by editing the slice below.

    Returns a manifest mapping prompt_name -> {parquet_path, tokens, shape}.
    """
    log("Step 3/8: extracting activations from base model")

    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    manifest: dict[str, dict[str, Any]] = {}
    for name, text in prompts.items():
        log(f"  prompt '{name}': {text[:60]}{'...' if len(text) > 60 else ''}")

        # Tokenise with truncation so long transcripts don't blow VRAM.
        enc = tok(text, return_tensors="pt",
                  truncation=True, max_length=max_tokens).to("cuda")
        seq_len = int(enc["input_ids"].shape[1])
        if seq_len == max_tokens:
            log(f"    note: truncated to {max_tokens} tokens (input was longer)")

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states[LAYER][0]  # [seq_len, d_model]

        if hs.shape[-1] != D_MODEL:
            die(f"unexpected hidden size {hs.shape[-1]} (expected {D_MODEL}). "
                f"Wrong base model or wrong layer index.")

        token_ids = enc["input_ids"][0].tolist()
        tokens = [tok.decode([t]) for t in token_ids]

        parquet_path = ACTS_DIR / f"{name}.parquet"
        table = pa.table({
            "token_index":       list(range(len(tokens))),
            "token_string":      tokens,
            "activation_vector": hs.float().cpu().tolist(),
        })
        pq.write_table(table, parquet_path)

        manifest[name] = {
            "parquet": str(parquet_path),
            "prompt":  text,
            "seq_len": int(hs.shape[0]),
            "tokens":  tokens,
        }
        log(f"    saved {parquet_path}  shape={tuple(hs.shape)}")

        # Per-prompt VRAM cleanup helps when iterating over many long sessions
        del out, hs, enc
        torch.cuda.empty_cache()

    # Free the base model before we touch the AV server
    del model
    import gc; gc.collect()
    torch.cuda.empty_cache()

    with open(RESULTS_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log("  manifest written")
    return manifest


# ---------------------------------------------------------------------------
# Step 4 + 5: launch SGLang and run nla_inference.py
# ---------------------------------------------------------------------------

def launch_sglang_av() -> subprocess.Popen:
    log("Step 4/8: launching SGLang server for the AV")
    if not port_is_free(SGLANG_PORT):
        die(f"port {SGLANG_PORT} is already in use. Free it or change SGLANG_PORT.")

    log_path = LOGS_DIR / "sglang_av.log"
    fh = open(log_path, "w")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", AV_MODEL,
            "--port", str(SGLANG_PORT),
            "--disable-radix-cache",
        ],
        stdout=fh, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    log(f"  PID {proc.pid}, logging to {log_path}")
    log("  waiting for server to come up (can take a few minutes)")
    if not wait_for_port(SGLANG_PORT, timeout=600):
        proc.terminate()
        die("SGLang did not come up within 10 minutes. Check the log file.")
    log("  server is up")
    return proc


def run_av_inference(manifest: dict[str, dict[str, Any]]) -> dict[str, Path]:
    log("Step 5/8: running NLA AV inference per prompt")
    out_paths: dict[str, Path] = {}
    for name, info in manifest.items():
        out_path = VERB_DIR / f"{name}.json"
        log(f"  verbalising {name}")
        cmd = [
            sys.executable, "nla_inference.py", AV_MODEL,
            "--sglang-url", f"http://localhost:{SGLANG_PORT}",
            "--parquet", info["parquet"],
        ]
        with open(out_path, "w") as fout, open(LOGS_DIR / f"av_{name}.log", "w") as ferr:
            rc = subprocess.call(cmd, stdout=fout, stderr=ferr)
        if rc != 0:
            log(f"    WARNING: nla_inference.py exited non-zero ({rc}) on {name}. "
                f"Check {LOGS_DIR / f'av_{name}.log'}.")
        out_paths[name] = out_path
    return out_paths


def shutdown_sglang(proc: subprocess.Popen) -> None:
    log("Step 6/8: shutting down SGLang AV server")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    # Give the GPU a moment to release memory
    time.sleep(5)
    import torch; torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Step 7: AR scoring
# ---------------------------------------------------------------------------

def load_verbalisations(path: Path) -> list[dict[str, Any]]:
    """nla_inference.py writes JSON; tolerate either a list-of-records or
    one JSON object per line."""
    text = path.read_text().strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        return [obj]
    except json.JSONDecodeError:
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records


def score_with_ar(manifest: dict[str, dict[str, Any]],
                  verb_paths: dict[str, Path]) -> list[dict[str, Any]]:
    """Reconstruct vectors from verbalisations via the AR and compute MSE
    against the original activations. Output is a flat list of records
    suitable for CSV export."""
    log("Step 7/8: AR reconstruction and MSE scoring")

    import torch
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer, AutoModel
    try:
        import yaml
    except ImportError:
        import importlib
        yaml = importlib.import_module("yaml")

    # Pull the AR sidecar so we know which token / layer to extract from.
    from huggingface_hub import hf_hub_download
    try:
        sidecar_path = hf_hub_download(AR_MODEL, "nla_meta.yaml")
        with open(sidecar_path) as f:
            sidecar = yaml.safe_load(f)
        log(f"  AR sidecar keys: {list(sidecar.keys())}")
    except Exception as e:
        log(f"  WARNING: could not load AR sidecar ({e}). "
            f"Falling back to last-token extraction.")
        sidecar = {}

    ar_tok = AutoTokenizer.from_pretrained(AR_MODEL)
    ar_model = AutoModel.from_pretrained(
        AR_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    ar_model.eval()

    def reconstruct(text: str) -> torch.Tensor:
        """Encode text through AR, take the final-token hidden state, project
        through the AR's linear head if present. The repo's nla.models.NLACriticModel
        is the canonical loader; if importable we use it, else we fall back
        to a generic AutoModel forward."""
        enc = ar_tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = ar_model(**enc, output_hidden_states=True)
            final_hidden = out.hidden_states[-1][0, -1]   # [d_model]
            # If the AR has a linear projection head, apply it. The convention
            # in the repo is a single Linear(d, d) named "head" or "projection".
            for attr in ("head", "projection", "linear"):
                if hasattr(ar_model, attr):
                    head = getattr(ar_model, attr)
                    if isinstance(head, torch.nn.Linear):
                        final_hidden = head(final_hidden)
                        break
        # L2-normalise per the paper convention
        return torch.nn.functional.normalize(final_hidden.float(), dim=-1).cpu()

    records: list[dict[str, Any]] = []
    for name, info in manifest.items():
        # Original activations
        table = pq.read_table(info["parquet"])
        original_vectors = torch.tensor(table["activation_vector"].to_pylist())
        original_norm    = torch.nn.functional.normalize(original_vectors.float(), dim=-1)
        tokens = info["tokens"]

        verbs = load_verbalisations(verb_paths[name])
        if not verbs:
            log(f"  WARNING: no verbalisations for {name}, skipping AR scoring")
            continue

        log(f"  scoring {name}: {len(verbs)} verbalisations against "
            f"{original_norm.shape[0]} activations")

        for v in verbs:
            # Be liberal in what we accept; nla_inference.py may name fields
            # 'verbalisation' or 'text' or 'explanation'.
            text = (v.get("verbalisation")
                    or v.get("verbalization")
                    or v.get("explanation")
                    or v.get("text")
                    or "")
            idx  = v.get("token_index", v.get("index"))
            if not text or idx is None or idx >= original_norm.shape[0]:
                continue

            reconstructed = reconstruct(text)
            target        = original_norm[idx]
            # MSE on L2-normalised vectors == 2(1 - cos)
            mse = float(((reconstructed - target) ** 2).mean())
            cos = float(torch.nn.functional.cosine_similarity(
                reconstructed.unsqueeze(0), target.unsqueeze(0)
            ).item())

            records.append({
                "prompt_name":   name,
                "token_index":   int(idx),
                "token":         tokens[idx] if idx < len(tokens) else "",
                "verbalisation": text,
                "mse":           round(mse, 6),
                "cosine":        round(cos, 6),
            })

    return records


# ---------------------------------------------------------------------------
# Step 8: write outputs and tarball
# ---------------------------------------------------------------------------

def write_outputs(records: list[dict[str, Any]]) -> None:
    log("Step 8/8: writing consolidated outputs")

    # JSON
    json_path = RESULTS_DIR / "ar_scores.json"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    log(f"  {json_path}")

    # CSV — for direct paste into the dissertation table
    import csv
    csv_path = RESULTS_DIR / "dissertation_table.csv"
    if records:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        log(f"  {csv_path}")
    else:
        log("  WARNING: no records to write to CSV")

    # Markdown summary — quick eyeball check before the scp
    md_path = RESULTS_DIR / "summary.md"
    with open(md_path, "w") as f:
        f.write("# NLA Demonstration Results\n\n")
        f.write(f"Base model: `{BASE_MODEL}` (layer {LAYER})\n")
        f.write(f"AV: `{AV_MODEL}`\n")
        f.write(f"AR: `{AR_MODEL}`\n\n")
        f.write(f"Total records: {len(records)}\n\n")
        if records:
            mses = [r["mse"] for r in records]
            f.write(f"MSE range: {min(mses):.4f} – {max(mses):.4f}\n")
            f.write(f"MSE mean:  {sum(mses)/len(mses):.4f}\n\n")
            f.write("## Sample rows\n\n")
            f.write("| prompt | token_idx | token | verbalisation (truncated) | mse | cos |\n")
            f.write("|---|---|---|---|---|---|\n")
            for r in records[:15]:
                verb_trunc = r["verbalisation"][:80].replace("|", "\\|")
                f.write(f"| {r['prompt_name']} | {r['token_index']} | "
                        f"`{r['token']}` | {verb_trunc} | {r['mse']:.4f} | {r['cosine']:.4f} |\n")
    log(f"  {md_path}")

    # Per-conversation report bundles — one self-contained markdown file per
    # session. Each report has the full conversation text, every token's
    # verbalisation, MSE/cosine per token, and basic stats. This is the
    # format intended for downstream review: open one file, read one session.
    reports_dir = RESULTS_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)

    manifest_path = RESULTS_DIR / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {}

    # Group records by session for fast per-session lookup
    by_session: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_session.setdefault(r["prompt_name"], []).append(r)

    for name, info in manifest.items():
        session_records = sorted(
            by_session.get(name, []),
            key=lambda r: r["token_index"]
        )

        report_path = reports_dir / f"{name}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# Session report: {name}\n\n")
            f.write(f"- Base model: `{BASE_MODEL}` (layer {LAYER})\n")
            f.write(f"- AV: `{AV_MODEL}`\n")
            f.write(f"- AR: `{AR_MODEL}`\n")
            f.write(f"- Token count: {info.get('seq_len', 'unknown')}\n")
            f.write(f"- Verbalised tokens: {len(session_records)}\n\n")

            if session_records:
                mses = [r["mse"] for r in session_records]
                cosines = [r["cosine"] for r in session_records]
                f.write("## Reconstruction statistics\n\n")
                f.write(f"- MSE: min {min(mses):.4f}, "
                        f"max {max(mses):.4f}, "
                        f"mean {sum(mses)/len(mses):.4f}\n")
                f.write(f"- Cosine: min {min(cosines):.4f}, "
                        f"max {max(cosines):.4f}, "
                        f"mean {sum(cosines)/len(cosines):.4f}\n\n")

                # Flag the tokens with the highest MSE — these are the
                # positions where verbalisation diverges most from the
                # original activation, often the most analytically
                # interesting points in the sequence.
                sorted_by_mse = sorted(
                    session_records, key=lambda r: -r["mse"]
                )[:10]
                f.write("## Top 10 highest-MSE tokens (potential signal points)\n\n")
                f.write("| token_idx | token | mse | cos | verbalisation |\n")
                f.write("|---|---|---|---|---|\n")
                for r in sorted_by_mse:
                    verb = r["verbalisation"].replace("|", "\\|").replace("\n", " ")
                    tok_display = r["token"].replace("|", "\\|").replace("\n", "\\n")
                    f.write(f"| {r['token_index']} | `{tok_display}` | "
                            f"{r['mse']:.4f} | {r['cosine']:.4f} | {verb[:120]} |\n")
                f.write("\n")

            f.write("## Full conversation text\n\n")
            f.write("```\n")
            prompt_text = info.get("prompt", "")
            # Cap the displayed prompt to keep reports readable; the full
            # text is preserved in manifest.json regardless.
            if len(prompt_text) > 20000:
                f.write(prompt_text[:20000])
                f.write(f"\n\n[... truncated, {len(prompt_text) - 20000} more chars in manifest.json ...]\n")
            else:
                f.write(prompt_text)
            f.write("\n```\n\n")

            f.write("## All verbalisations (in token order)\n\n")
            f.write("| token_idx | token | mse | cos | verbalisation |\n")
            f.write("|---|---|---|---|---|\n")
            for r in session_records:
                verb = r["verbalisation"].replace("|", "\\|").replace("\n", " ")
                tok_display = r["token"].replace("|", "\\|").replace("\n", "\\n")
                f.write(f"| {r['token_index']} | `{tok_display}` | "
                        f"{r['mse']:.4f} | {r['cosine']:.4f} | {verb[:200]} |\n")

    log(f"  wrote {len(manifest)} per-session reports to {reports_dir}")


def tarball_results() -> None:
    tar_path = RESULTS_DIR.parent / "nla_results.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(RESULTS_DIR, arcname="results")
    size_mb = tar_path.stat().st_size / 1024**2
    log(f"  tarball: {tar_path} ({size_mb:.1f} MB)")
    log("")
    log("Done. To copy off the pod:")
    log(f"    scp -P <port> root@<pod-ip>:{tar_path.resolve()} .")
    log("")
    log("Then terminate (do not pause) the pod from the RunPod dashboard.")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _csv_file_to_text(path: Path) -> str:
    """Read a CSV containing one conversation and return it as a single
    prompt string for activation extraction.

    Two modes, auto-detected:

    1. **Database-dump schema** (one row, columns include `messages_json`
       holding a JSON array of {role, content} turn objects). The loader
       parses messages_json and reconstructs the conversation as
       role-tagged turns, ignoring the verbose token_probabilities and
       embedded payload blobs.

    2. **Generic CSV** (anything else). Falls back to concatenating every
       non-empty cell in row order — schema-agnostic last resort.

    The reconstructed prompt looks like:

        [SYSTEM]
        You are a helpful assistant.

        [USER]
        <user turn 1>

        [ASSISTANT]
        <assistant turn 1>

    This format keeps role boundaries visible to the model without
    relying on any specific chat template.
    """
    import csv as _csv

    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = _csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    # --- Mode 1: database-dump schema ---
    if "messages_json" in fieldnames and rows:
        row = rows[0]  # one conversation per file
        raw = row.get("messages_json", "")
        if raw:
            try:
                messages = json.loads(raw)
            except json.JSONDecodeError:
                messages = None
            if isinstance(messages, list) and messages:
                parts: list[str] = []
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    role = str(msg.get("role", "")).strip().upper() or "TURN"
                    content = str(msg.get("content", "")).strip()
                    if not content:
                        continue
                    parts.append(f"[{role}]\n{content}")
                if parts:
                    return "\n\n".join(parts)

    # --- Mode 2: generic fallback ---
    parts: list[str] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = _csv.reader(f)
        for r in reader:
            for cell in r:
                cell = (cell or "").strip()
                if cell:
                    parts.append(cell)
    return "\n".join(parts)


def load_prompts(args) -> dict[str, str]:
    if args.prompts_dir:
        directory = Path(args.prompts_dir)
        if not directory.is_dir():
            die(f"--prompts-dir {directory} is not a directory")
        csv_files = sorted(directory.glob("*.csv"))
        if not csv_files:
            die(f"no .csv files found in {directory}")
        prompts: dict[str, str] = {}
        for path in csv_files:
            try:
                text = _csv_file_to_text(path)
            except Exception as e:
                log(f"  WARNING: failed to read {path.name}: {e}")
                continue
            if not text:
                log(f"  WARNING: {path.name} produced empty text, skipping")
                continue
            name = path.stem
            prompts[name] = text
        if not prompts:
            die(f"no usable CSV files in {directory}")
        log(f"loaded {len(prompts)} conversations from {directory}")
        return prompts

    if args.prompts_csv:
        import csv as _csv
        prompts: dict[str, str] = {}
        with open(args.prompts_csv, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            if "name" not in reader.fieldnames or "prompt" not in reader.fieldnames:
                die("--prompts-csv must have columns 'name' and 'prompt'")
            for row in reader:
                name = row["name"].strip()
                prompt = row["prompt"]
                if not name or not prompt:
                    continue
                prompts[name] = prompt
        if not prompts:
            die(f"no usable rows in {args.prompts_csv}")
        log(f"loaded {len(prompts)} prompts from CSV")
        return prompts

    if args.prompts_file:
        with open(args.prompts_file) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            die(f"--prompts-file must contain a JSON object of {{name: prompt}}")
        return data
    return DEFAULT_PROMPTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip pre-download (weights already cached)")
    parser.add_argument("--extract-only", action="store_true",
                        help="Run only steps 1–3 (no server, no AR scoring)")
    parser.add_argument("--score-only", action="store_true",
                        help="Skip extraction; reuse existing activations + verbalisations")
    parser.add_argument("--prompts-file", type=str, default=None,
                        help="Path to JSON file with {name: prompt} mapping")
    parser.add_argument("--prompts-csv", type=str, default=None,
                        help="Path to CSV file with 'name,prompt' columns. "
                             "Each row's prompt may be an entire multi-turn "
                             "conversation; the script extracts activations "
                             "over the whole sequence.")
    parser.add_argument("--prompts-dir", type=str, default=None,
                        help="Path to a directory containing one CSV file per "
                             "conversation. Each CSV is read as a whole and "
                             "its filename (minus .csv) is used as the prompt "
                             "name. Schema-agnostic: all non-empty cells are "
                             "concatenated in row order.")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Truncate any prompt longer than this many "
                             "tokens before extraction (default: 8192). "
                             "Qwen2.5-7B supports 32k but longer sequences "
                             "use more VRAM during the forward pass.")
    args = parser.parse_args()

    ensure_dirs()
    check_environment()

    if not args.skip_download and not args.score_only:
        download_weights()

    prompts = load_prompts(args)

    if args.score_only:
        manifest_path = RESULTS_DIR / "manifest.json"
        if not manifest_path.exists():
            die("--score-only requires results/manifest.json from a prior run")
        with open(manifest_path) as f:
            manifest = json.load(f)
        verb_paths = {name: VERB_DIR / f"{name}.json" for name in manifest}
    else:
        manifest = extract_activations(prompts, max_tokens=args.max_tokens)
        if args.extract_only:
            log("--extract-only set; stopping after activation extraction.")
            return
        server = launch_sglang_av()
        try:
            verb_paths = run_av_inference(manifest)
        finally:
            shutdown_sglang(server)

    records = score_with_ar(manifest, verb_paths)
    write_outputs(records)
    tarball_results()


if __name__ == "__main__":
    main()
