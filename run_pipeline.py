#!/usr/bin/env python3
"""
End-to-end NLA pipeline for the Dr House session.

Runs on a fresh RunPod H100. Handles:
  1. Dependency setup (transformers version, eager attention, pyarrow)
  2. HF cache configuration on /workspace
  3. Model downloads (Qwen2.5-7B + kitft AV/AR checkpoints)
  4. Activation extraction at layer 20
  5. AV verbalisation (one description per token)
  6. AR scoring (MSE + cosine reconstruction)
  7. Reproducibility analysis (mirrors v6 chapter findings)

Usage:
    python run_pipeline.py --csv path/to/dr_house.csv --out-dir ./results/

Expected runtime: ~30-40 min on H100 80GB for an 8192-token session.

Inputs:
    --csv         Path to the conversation CSV (single session)
    --out-dir     Where to write outputs (default: ./results/)
    --max-tokens  Cap on tokens to process (default: 8192)
    --layer       Layer to probe (default: 20)
    --batch-size  AV inference batch size (default: 4)

Outputs (all in --out-dir):
    activations.parquet          [n_tokens, 3584] layer-20 residual stream
    verbalisations.json          Per-token AV descriptions
    ar_scores.json               Per-token MSE + cosine
    analysis_report.md           Chapter-ready quantitative findings
    tables.csv                   Per-category position-stratified stats
    verbalisation_samples.csv    Exemplar verbalisations for cleared categories
    fve_results.json             FVE numbers vs paper's 0.6-0.8 range
"""

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path


# ============================================================================
# Stage 1: Dependency setup
# ============================================================================

def setup_environment(workspace="/workspace"):
    """Install dependencies, configure HF cache, clean up CUDA conflicts.

    This handles the dependency wrangling we worked through over the course
    of the original session: transformers version pinning, eager attention
    (since flash-attn needs CUDA matching the deep_gemm install), and HF
    cache on the workspace volume so checkpoints persist across pod restarts.
    """
    print("=" * 70)
    print("Stage 1: Environment setup")
    print("=" * 70)

    # HF cache on workspace so checkpoints persist
    os.environ["HF_HOME"] = f"{workspace}/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = f"{workspace}/hf_cache"
    Path(f"{workspace}/hf_cache").mkdir(parents=True, exist_ok=True)
    print(f"  HF cache: {os.environ['HF_HOME']}")

    # Install pinned dependencies
    packages = [
        "transformers==4.50.0",
        "accelerate>=0.26.0",
        "torch",
        "numpy",
        "pyarrow",
        "safetensors",
        "tqdm",
        "pandas",
    ]
    print("  Installing dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "--break-system-packages"] + packages,
        check=True,
    )

    # Remove deep_gemm if present (causes libnvrtc.so.13 CUDA mismatch)
    deep_gemm_path = Path("/usr/local/lib/python3.11/dist-packages/deep_gemm")
    if deep_gemm_path.exists():
        print("  Removing deep_gemm (CUDA version mismatch)...")
        subprocess.run(["rm", "-rf", str(deep_gemm_path)], check=False)

    print("  Done.")


# ============================================================================
# Stage 2: Conversation loading
# ============================================================================

def load_conversation_csv(csv_path: Path, tokenizer, max_tokens: int = 8192):
    """Load CSV and tokenize the full conversation.

    CSV schema (from your existing pipeline):
        Multiple rows per session, one row per turn.
        Columns: role, content, ... (other metadata)

    Returns: token_ids tensor, token_strings list.
    """
    print(f"\nStage 2: Loading {csv_path}")
    import csv as csv_mod
    csv_mod.field_size_limit(2**31 - 1)

    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"  Loaded {len(rows)} rows")

    # Build conversation in Qwen chat format
    messages = []
    for row in rows:
        role = (row.get("role") or row.get("speaker") or "user").lower()
        content = row.get("content") or row.get("message") or ""
        if not content:
            continue
        if role in ("system", "user", "assistant"):
            messages.append({"role": role, "content": content})

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    encoded = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_tokens
    )
    token_ids = encoded["input_ids"][0]
    token_strings = [tokenizer.decode([t]) for t in token_ids]
    print(f"  Tokenized to {len(token_ids)} tokens")
    return token_ids, token_strings


# ============================================================================
# Stage 3: Activation extraction
# ============================================================================

def extract_activations(token_ids, layer: int = 20):
    """Run Qwen2.5-7B forward pass and capture layer-l residual stream."""
    print(f"\nStage 3: Activation extraction (layer {layer})")
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    captured = {}

    def hook(module, inp, out):
        captured["acts"] = out[0].detach().cpu().float()

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model(token_ids.unsqueeze(0).to(model.device))
    finally:
        handle.remove()

    acts = captured["acts"][0]  # [n_tokens, 3584]
    print(f"  Activations shape: {acts.shape}")

    # Free target model memory before loading AV
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return acts


def save_activations_parquet(acts, out_path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    print(f"  Writing {out_path}")
    table = pa.table({
        "token_index": list(range(acts.shape[0])),
        "activation_vector": [acts[i].tolist() for i in range(acts.shape[0])],
    })
    pq.write_table(table, out_path)


# ============================================================================
# Stage 4: AV verbalisation
# ============================================================================

AV_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "You are an expert at describing the concept represented by an "
    "activation vector from a language model.<|im_end|>\n"
    "<|im_start|>user\n"
    "Describe the concept represented by this activation: <ACT><|im_end|>\n"
    "<|im_start|>assistant\n"
)


def run_av(acts, batch_size: int = 4, max_new_tokens: int = 80,
           injection_scale: float = 1.0):
    """Generate AV verbalisations using kitft/nla-qwen2.5-7b-L20-av.

    Replaces the embedding of a special <ACT> token with the (scaled)
    activation vector at each position, then samples a natural-language
    description.
    """
    print(f"\nStage 4: AV verbalisation (batch={batch_size})")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    tok = AutoTokenizer.from_pretrained("kitft/nla-qwen2.5-7b-L20-av")
    av = AutoModelForCausalLM.from_pretrained(
        "kitft/nla-qwen2.5-7b-L20-av",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    av.eval()

    # Build prompt with placeholder token
    prefix, suffix = AV_PROMPT_TEMPLATE.split("<ACT>")
    prefix_ids = tok(prefix, return_tensors="pt").input_ids[0]
    suffix_ids = tok(suffix, return_tensors="pt").input_ids[0]
    placeholder_id = tok.pad_token_id or tok.eos_token_id

    n_tokens = acts.shape[0]
    verbalisations = []

    embed = av.get_input_embeddings()

    for batch_start in tqdm(range(0, n_tokens, batch_size),
                            desc="  AV inference"):
        batch_end = min(batch_start + batch_size, n_tokens)
        batch_acts = acts[batch_start:batch_end].to(av.device).to(av.dtype)

        # Build inputs_embeds: [B, T, D] = [B, prefix + 1 + suffix, D]
        prefix_emb = embed(prefix_ids.to(av.device))  # [P, D]
        suffix_emb = embed(suffix_ids.to(av.device))  # [S, D]

        batch_inputs = []
        for i in range(batch_acts.shape[0]):
            act_emb = batch_acts[i].unsqueeze(0) * injection_scale  # [1, D]
            combined = torch.cat([prefix_emb, act_emb, suffix_emb], dim=0)
            batch_inputs.append(combined)
        batch_emb = torch.stack(batch_inputs)  # [B, T, D]

        attn_mask = torch.ones(
            batch_emb.shape[:2], dtype=torch.long, device=av.device
        )

        with torch.no_grad():
            out = av.generate(
                inputs_embeds=batch_emb,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=placeholder_id,
            )

        # out only contains generated tokens when using inputs_embeds
        for i in range(out.shape[0]):
            text = tok.decode(out[i], skip_special_tokens=True).strip()
            verbalisations.append({
                "token_index": batch_start + i,
                "verbalisation": text,
            })

    del av
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  Generated {len(verbalisations)} verbalisations")
    return verbalisations


# ============================================================================
# Stage 5: AR scoring
# ============================================================================

AR_PROMPT_TEMPLATE = (
    "<|im_start|>user\n"
    "Reconstruct the activation from this description: <DESC><|im_end|>\n"
    "<|im_start|>assistant\n"
)


def run_ar(acts, verbalisations, layer: int = 20, batch_size: int = 4):
    """Score reconstruction quality using kitft/nla-qwen2.5-7b-L20-ar.

    AR is the target model truncated to first L layers + a learned affine
    map at the final token. Returns MSE and cosine per token.
    """
    print(f"\nStage 5: AR scoring (batch={batch_size})")
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    tok = AutoTokenizer.from_pretrained("kitft/nla-qwen2.5-7b-L20-ar")
    ar = AutoModelForCausalLM.from_pretrained(
        "kitft/nla-qwen2.5-7b-L20-ar",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        output_hidden_states=True,
    )
    ar.eval()

    # Unit-normalise reference activations (paper convention)
    acts_n = F.normalize(acts, p=2, dim=-1)

    scores = []
    for v in tqdm(verbalisations, desc="  AR inference"):
        idx = v["token_index"]
        prompt = AR_PROMPT_TEMPLATE.replace("<DESC>", v["verbalisation"])
        enc = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=512).to(ar.device)

        with torch.no_grad():
            out = ar(**enc, output_hidden_states=True)

        # Layer-l hidden state at final token
        hidden = out.hidden_states[layer][0, -1, :].cpu().float()
        hidden_n = F.normalize(hidden, p=2, dim=-1)
        ref_n = acts_n[idx]

        mse = ((hidden_n - ref_n) ** 2).mean().item()
        cosine = (hidden_n * ref_n).sum().item()

        scores.append({
            "token_index": idx,
            "mse": mse,
            "cosine": cosine,
        })

    del ar
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return scores


# ============================================================================
# Stage 6: Reproducibility analysis (mirrors v6 chapter)
# ============================================================================

PRE_ATTACK_END = 2156
ATTACK_START = 2158
D_MODEL = 3584

DECISION_CATEGORIES = {
    "hard_refusal": [
        "sorry", "can't", "cannot", "won't", "unable", "refuse", "decline",
    ],
    "risk_vocabulary": [
        "harmful", "illegal", "dangerous", "unethical",
        "inappropriate", "risk", "concern", "safety",
    ],
    "epistemic_hedging": [
        "perhaps", "maybe", "might", "could",
        "possibly", "probably", "seem",
    ],
    "meta_cognition": [
        "ethical", "appropriate", "policy", "guidelines",
        "should", "must", "compliance",
    ],
    "directive_against": [
        "avoid", "prevent", "block", "stop", "halt", "cease", "reject",
    ],
    "hedging_pivot": [
        "but", "however", "although", "wait", "actually",
        "instead", "rather",
    ],
    "condition_markers": [
        " if ", " unless ", " when ", " whether ", " assuming", " given",
    ],
    "compliance_pivot": [
        "comply", "follow", "obey", "accept",
        "agree", "proceed", "continue",
    ],
    "meta_speech": [
        " say", " said", " tell", " told", " explain", " describe", " provide",
    ],
    "internal_deliberation": [
        "wait", "hmm", "actually", "reconsider", "rethink", "pause",
    ],
}

CLEARED_SAMPLE_CATEGORIES = [
    "hard_refusal", "risk_vocabulary", "internal_deliberation",
    "epistemic_hedging", "directive_against", "compliance_pivot",
]

REAGENT_TOKENS = ["aluminum", "lithium", "ethanol"]


def bootstrap_ci(values, n_iter=10000, alpha=0.05, seed=42):
    if len(values) < 2:
        return (statistics.mean(values) if values else None, None, None)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_iter):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(n_iter * alpha / 2)]
    hi = means[int(n_iter * (1 - alpha / 2))]
    return (statistics.mean(values), lo, hi)


def run_analysis(scores, verbalisations, token_strings, activations_path,
                 out_dir: Path):
    """All quantitative analyses from v6 chapter."""
    print("\nStage 6: Reproducibility analysis")
    import numpy as np

    # Combine scores with token strings
    full = []
    verb_by_idx = {v["token_index"]: v["verbalisation"] for v in verbalisations}
    for s in scores:
        idx = s["token_index"]
        full.append({
            "token_index": idx,
            "token": token_strings[idx] if idx < len(token_strings) else "",
            "mse": s["mse"],
            "cosine": s["cosine"],
            "verbalisation": verb_by_idx.get(idx, ""),
        })

    # Save the per-token detail JSON for inspection
    with open(out_dir / "ar_scores.json", "w") as f:
        json.dump(full, f, indent=2)
    print(f"  Wrote ar_scores.json ({len(full)} tokens)")

    # Session-level stats
    mses = [s["mse"] for s in full]
    cosines = [s["cosine"] for s in full]
    session_stats = {
        "n_tokens": len(full),
        "mse": {
            "min": min(mses), "mean": statistics.mean(mses),
            "max": max(mses), "stdev": statistics.stdev(mses),
            "p10": float(np.percentile(mses, 10)),
            "p50": float(np.percentile(mses, 50)),
            "p90": float(np.percentile(mses, 90)),
            "p99": float(np.percentile(mses, 99)),
        },
        "cosine": {
            "min": min(cosines), "mean": statistics.mean(cosines),
            "max": max(cosines), "stdev": statistics.stdev(cosines),
            "p1": float(np.percentile(cosines, 1)),
            "p10": float(np.percentile(cosines, 10)),
            "p50": float(np.percentile(cosines, 50)),
            "p90": float(np.percentile(cosines, 90)),
            "p99": float(np.percentile(cosines, 99)),
        },
    }

    # Region split
    pre = [s["cosine"] for s in full if s["token_index"] < PRE_ATTACK_END]
    att = [s["cosine"] for s in full if s["token_index"] >= ATTACK_START]
    pre_mean, pre_lo, pre_hi = bootstrap_ci(pre)
    att_mean, att_lo, att_hi = bootstrap_ci(att)
    region_split = {
        "pre_attack": {"n": len(pre), "mean": pre_mean,
                       "ci": [pre_lo, pre_hi]},
        "during_attack": {"n": len(att), "mean": att_mean,
                          "ci": [att_lo, att_hi]},
    }

    # Per-category position-stratified analysis
    categories = {}
    for cat_name, keywords in DECISION_CATEGORIES.items():
        c_pre, c_att = [], []
        for s in full:
            tok = s["token"].strip().lower()
            if not tok or len(tok) > 25:
                continue
            if not any(kw.lower().strip() in tok for kw in keywords):
                continue
            if s["token_index"] < PRE_ATTACK_END:
                c_pre.append(s["cosine"])
            elif s["token_index"] >= ATTACK_START:
                c_att.append(s["cosine"])
        p_mean, p_lo, p_hi = bootstrap_ci(c_pre)
        a_mean, a_lo, a_hi = bootstrap_ci(c_att)
        categories[cat_name] = {
            "pre_n": len(c_pre), "pre_mean": p_mean, "pre_ci": [p_lo, p_hi],
            "att_n": len(c_att), "att_mean": a_mean, "att_ci": [a_lo, a_hi],
            "delta": (a_mean - p_mean) if (p_mean and a_mean) else None,
        }

    # Reagent reconstruction (scores only, no verbalisations surfaced)
    reagent_rows = []
    for s in full:
        tok = s["token"].strip().lower()
        if tok in REAGENT_TOKENS:
            reagent_rows.append({
                "token_index": s["token_index"], "token": tok,
                "mse": s["mse"], "cosine": s["cosine"],
            })
    reagent_rows.sort(key=lambda x: x["cosine"])

    # FVE computation
    fve = None
    if activations_path and activations_path.exists():
        import pyarrow.parquet as pq
        table = pq.read_table(activations_path)
        acts = np.array(table["activation_vector"].to_pylist())
        norms = np.linalg.norm(acts, axis=1, keepdims=True)
        acts_n = acts / norms
        mses_arr = np.array([s["mse"] for s in full])
        sq_errors = mses_arr * D_MODEL
        mean_h = acts_n.mean(axis=0)
        sq_devs = ((acts_n - mean_h) ** 2).sum(axis=1)
        fve_total = 1 - sq_errors.sum() / sq_devs.sum()
        pre_mask = np.array([s["token_index"] < PRE_ATTACK_END
                             for s in full])
        att_mask = np.array([s["token_index"] >= ATTACK_START
                             for s in full])
        fve_pre = 1 - sq_errors[pre_mask].sum() / sq_devs[pre_mask].sum()
        fve_att = 1 - sq_errors[att_mask].sum() / sq_devs[att_mask].sum()
        fve = {
            "session_fve": float(fve_total),
            "pre_attack_fve": float(fve_pre),
            "attack_region_fve": float(fve_att),
            "paper_reference_range": [0.6, 0.8],
        }
        with open(out_dir / "fve_results.json", "w") as f:
            json.dump(fve, f, indent=2)

    # Write markdown report
    write_report(session_stats, region_split, categories, reagent_rows,
                 fve, out_dir / "analysis_report.md")

    # Write CSV tables
    write_tables_csv(region_split, categories, out_dir / "tables.csv")

    # Write exemplar verbalisations for cleared categories
    write_samples_csv(full, out_dir / "verbalisation_samples.csv")

    print("  Analysis complete.")


def write_report(session_stats, region_split, categories, reagent_rows,
                 fve, out_path):
    lines = []
    lines.append("# Dr House Session — Pipeline Output\n")
    lines.append(
        "Mirrors v6 chapter findings. Citation: Fraser-Taliente et al. "
        "(2026), https://transformer-circuits.pub/2026/nla/\n"
    )

    ss = session_stats
    lines.append("## 1. Session-level statistics\n")
    lines.append(f"Tokens: {ss['n_tokens']}\n")
    lines.append(f"MSE: mean {ss['mse']['mean']:.6f}, "
                 f"stdev {ss['mse']['stdev']:.6f}, "
                 f"range [{ss['mse']['min']:.6f}, {ss['mse']['max']:.6f}]\n")
    lines.append(f"Cosine: mean {ss['cosine']['mean']:.4f}, "
                 f"stdev {ss['cosine']['stdev']:.4f}, "
                 f"range [{ss['cosine']['min']:.4f}, "
                 f"{ss['cosine']['max']:.4f}]\n")

    rs = region_split
    lines.append("## 2. Region split\n")
    lines.append(f"- Pre-attack (n={rs['pre_attack']['n']}): "
                 f"cos {rs['pre_attack']['mean']:.4f} "
                 f"[{rs['pre_attack']['ci'][0]:.4f}, "
                 f"{rs['pre_attack']['ci'][1]:.4f}]")
    lines.append(f"- During attack (n={rs['during_attack']['n']}): "
                 f"cos {rs['during_attack']['mean']:.4f} "
                 f"[{rs['during_attack']['ci'][0]:.4f}, "
                 f"{rs['during_attack']['ci'][1]:.4f}]\n")

    lines.append("## 3. Eleven-category position-stratified analysis\n")
    lines.append("| Category | Pre n | Pre cos | Att n | Att cos | Delta |")
    lines.append("|---|---|---|---|---|---|")
    for cat_name, info in categories.items():
        pre_str = f"{info['pre_mean']:.3f}" if info["pre_mean"] else "—"
        att_str = f"{info['att_mean']:.3f}" if info["att_mean"] else "—"
        del_str = f"{info['delta']:+.3f}" if info["delta"] else "—"
        lines.append(f"| {cat_name} | {info['pre_n']} | {pre_str} | "
                     f"{info['att_n']} | {att_str} | {del_str} |")
    lines.append("")

    lines.append("## 4. Reagent reconstruction scores\n")
    lines.append("| idx | token | MSE | Cosine |")
    lines.append("|---|---|---|---|")
    for r in reagent_rows:
        lines.append(f"| {r['token_index']} | {r['token']} | "
                     f"{r['mse']:.6f} | {r['cosine']:.4f} |")
    lines.append("")

    if fve:
        lines.append("## 5. FVE\n")
        lines.append(f"- Session FVE: {fve['session_fve']:.4f}")
        lines.append(f"- Pre-attack FVE: {fve['pre_attack_fve']:.4f}")
        lines.append(f"- Attack region FVE: {fve['attack_region_fve']:.4f}")
        lines.append(f"- Paper's trained-NLA range: "
                     f"{fve['paper_reference_range']}\n")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def write_tables_csv(region_split, categories, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "region", "n", "mean_cos", "ci_lo", "ci_hi"])
        rs = region_split
        for region_label, key in [("pre_attack", "pre_attack"),
                                  ("during_attack", "during_attack")]:
            r = rs[key]
            ci = r.get("ci") or [None, None]
            w.writerow(["SESSION", region_label, r["n"],
                        f"{r['mean']:.4f}" if r["mean"] else "",
                        f"{ci[0]:.4f}" if ci[0] is not None else "",
                        f"{ci[1]:.4f}" if ci[1] is not None else ""])
        for cat_name, info in categories.items():
            for region_label, n_key, m_key, ci_key in [
                ("pre_attack", "pre_n", "pre_mean", "pre_ci"),
                ("during_attack", "att_n", "att_mean", "att_ci"),
            ]:
                n = info[n_key]
                m = info[m_key]
                ci = info.get(ci_key) or [None, None]
                w.writerow([cat_name, region_label, n,
                            f"{m:.4f}" if m else "",
                            f"{ci[0]:.4f}" if ci[0] is not None else "",
                            f"{ci[1]:.4f}" if ci[1] is not None else ""])


def write_samples_csv(full, out_path):
    """Exemplar verbalisations for cleared methodology-illustration categories.
    Pulls up to 3 lowest-cosine tokens per category."""
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "category", "token_index", "token", "region",
            "mse", "cosine", "verbalisation",
        ])
        w.writeheader()
        for cat_name in CLEARED_SAMPLE_CATEGORIES:
            keywords = DECISION_CATEGORIES[cat_name]
            cat_rows = []
            for s in full:
                tok = s["token"].strip().lower()
                if not tok or len(tok) > 25:
                    continue
                if not any(kw.lower().strip() in tok for kw in keywords):
                    continue
                cat_rows.append({
                    "category": cat_name,
                    "token_index": s["token_index"],
                    "token": s["token"],
                    "region": ("pre-attack"
                               if s["token_index"] < PRE_ATTACK_END
                               else "during attack"),
                    "mse": f"{s['mse']:.6f}",
                    "cosine": f"{s['cosine']:.4f}",
                    "verbalisation": s["verbalisation"][:300],
                })
            cat_rows.sort(key=lambda x: float(x["cosine"]))
            for row in cat_rows[:3]:
                w.writerow(row)


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True,
                    help="Dr House session CSV")
    ap.add_argument("--out-dir", type=Path, default=Path("./results"))
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--skip-setup", action="store_true",
                    help="Skip dependency install (already done)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_setup:
        setup_environment()

    # Import after setup so dependencies are available
    from transformers import AutoTokenizer
    import torch

    print("\nStage 2-prep: Loading tokenizer")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    token_ids, token_strings = load_conversation_csv(
        args.csv, tok, max_tokens=args.max_tokens
    )

    acts = extract_activations(token_ids, layer=args.layer)
    activations_path = args.out_dir / "activations.parquet"
    save_activations_parquet(acts, activations_path)

    verbalisations = run_av(acts, batch_size=args.batch_size)
    with open(args.out_dir / "verbalisations.json", "w") as f:
        json.dump(verbalisations, f, indent=2)

    scores = run_ar(acts, verbalisations, layer=args.layer,
                    batch_size=args.batch_size)

    run_analysis(scores, verbalisations, token_strings,
                 activations_path, args.out_dir)

    print("\n" + "=" * 70)
    print(f"DONE. Outputs in {args.out_dir}/")
    print("=" * 70)
    print("  activations.parquet")
    print("  verbalisations.json")
    print("  ar_scores.json")
    print("  analysis_report.md")
    print("  tables.csv")
    print("  verbalisation_samples.csv")
    print("  fve_results.json")


if __name__ == "__main__":
    main()
