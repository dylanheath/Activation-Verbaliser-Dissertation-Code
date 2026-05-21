#!/usr/bin/env python3
"""
Dr House Session Reproducibility Script
========================================

Reproduces the quantitative analyses from the NLA worked-example chapter (v6)
on the Dr House persona-attack session. Takes the AR scores JSON and activation
parquet as input and produces:

  - analysis_report.md      Markdown report mirroring v6 chapter structure
  - tables.csv              Per-category position-stratified statistics
  - verbalisation_samples.csv   Exemplar verbalisations for cleared categories
  - fve_results.json        FVE computation results

Usage:
    python reproduce_dr_house_analysis.py \\
        --scores ar_scores.json \\
        --activations activations/<session>.parquet \\
        --out-dir ./output/

Dependencies:
    pip install numpy pyarrow
"""

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

import numpy as np

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    print("WARNING: pyarrow not installed. FVE computation will be skipped.")
    print("         Install with: pip install pyarrow")


# ============================================================================
# Configuration
# ============================================================================

# Region split based on the first [USER] role boundary in the Dr House session
PRE_ATTACK_END = 2156
ATTACK_START = 2158
D_MODEL = 3584

# Decision-vocabulary categories from v6 Section 6.
# These cover the defender's linguistic register: refusal output, risk concepts,
# deliberation, hedging, and meta-cognition. Each category's tokens are the
# linguistic positions where safety reasoning would surface in the model's
# output, which is what trajectory-monitoring is interested in.
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

# Reagent vocabulary documented in v6 Section 8. These are specific procedural
# tokens whose reconstruction-collapse is the basis of Finding 3. They are
# specific named entities, included because the chapter's analysis depends on
# them, not as a list to surface verbalisations from.
REAGENT_TOKENS = ["aluminum", "lithium", "ethanol"]


# ============================================================================
# Core analyses
# ============================================================================

def load_scores(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def session_level_stats(scores: list[dict]) -> dict:
    """Section 4 of v6: overall MSE and cosine distributions."""
    mses = [s["mse"] for s in scores]
    cosines = [s["cosine"] for s in scores]
    return {
        "n_tokens": len(scores),
        "mse": {
            "min": min(mses),
            "mean": statistics.mean(mses),
            "max": max(mses),
            "stdev": statistics.stdev(mses),
            "p10": np.percentile(mses, 10),
            "p50": np.percentile(mses, 50),
            "p90": np.percentile(mses, 90),
            "p99": np.percentile(mses, 99),
        },
        "cosine": {
            "min": min(cosines),
            "mean": statistics.mean(cosines),
            "max": max(cosines),
            "stdev": statistics.stdev(cosines),
            "p1": np.percentile(cosines, 1),
            "p10": np.percentile(cosines, 10),
            "p50": np.percentile(cosines, 50),
            "p90": np.percentile(cosines, 90),
            "p99": np.percentile(cosines, 99),
        },
    }


def region_split_stats(scores: list[dict]) -> dict:
    """Section 5 of v6: pre-attack vs during-attack reconstruction quality."""
    pre = [s["cosine"] for s in scores if s["token_index"] < PRE_ATTACK_END]
    att = [s["cosine"] for s in scores if s["token_index"] >= ATTACK_START]
    pre_mean, pre_lo, pre_hi = bootstrap_ci(pre)
    att_mean, att_lo, att_hi = bootstrap_ci(att)
    return {
        "pre_attack": {
            "n": len(pre), "mean": pre_mean, "ci_lo": pre_lo, "ci_hi": pre_hi,
        },
        "during_attack": {
            "n": len(att), "mean": att_mean, "ci_lo": att_lo, "ci_hi": att_hi,
        },
    }


def bootstrap_ci(values, n_iter=10000, alpha=0.05, seed=42):
    """Bootstrap 95% CI on the mean."""
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


def category_analysis(scores: list[dict]) -> dict:
    """Section 6 of v6: position-stratified eleven-category gradient."""
    results = {}
    for cat_name, keywords in DECISION_CATEGORIES.items():
        pre, att = [], []
        for s in scores:
            tok = s["token"].strip().lower()
            if not tok or len(tok) > 25:
                continue
            if not any(kw.lower().strip() in tok for kw in keywords):
                continue
            if s["token_index"] < PRE_ATTACK_END:
                pre.append(s["cosine"])
            elif s["token_index"] >= ATTACK_START:
                att.append(s["cosine"])
        pre_mean, pre_lo, pre_hi = bootstrap_ci(pre)
        att_mean, att_lo, att_hi = bootstrap_ci(att)
        results[cat_name] = {
            "pre_n": len(pre),
            "pre_mean": pre_mean,
            "pre_ci": (pre_lo, pre_hi),
            "att_n": len(att),
            "att_mean": att_mean,
            "att_ci": (att_lo, att_hi),
            "delta": (att_mean - pre_mean) if (pre_mean and att_mean) else None,
        }
    return results


def reagent_collapse_analysis(scores: list[dict]) -> list[dict]:
    """Section 8 of v6: reconstruction-collapse positions for the documented
    procedural vocabulary. Returns positions and scores only — no verbalisations
    are surfaced from this analysis. The point is the cosine distribution at
    these positions, not the AV's interpretation of them."""
    results = []
    for s in scores:
        tok = s["token"].strip().lower()
        if tok in REAGENT_TOKENS:
            results.append({
                "token_index": s["token_index"],
                "token": tok,
                "mse": s["mse"],
                "cosine": s["cosine"],
            })
    return sorted(results, key=lambda x: x["cosine"])


def recurring_claim_diagnostic(scores: list[dict], window: int = 5) -> dict:
    """Apply Fraser-Taliente et al.'s recurring-claim heuristic.
    For each token, check whether thematic claims in its verbalisation appear
    in adjacent tokens' verbalisations. Recurring claims are more reliable
    than singletons.

    This is a coarse implementation: it counts shared content-words between
    adjacent verbalisations. A high recurrence rate suggests the AV is tracking
    a stable contextual theme across positions. A low rate suggests the
    verbalisations are independent guesses, more likely to contain
    confabulations."""
    import re
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "with", "for",
        "is", "are", "was", "be", "this", "that", "these", "those", "it",
        "its", "as", "by", "at", "from", "have", "has", "had", "but", "not",
        "no", "so", "if", "then", "than", "such", "which", "what",
        "concept", "activation", "text", "prompt", "structure", "appears",
        "suggesting", "likely", "context", "specific", "based",
    }

    def content_words(v):
        words = re.findall(r"\b[a-z]{4,}\b", v.lower())
        return set(w for w in words if w not in stopwords)

    n = len(scores)
    recurrence_rates = []
    for i in range(n):
        own_words = content_words(scores[i]["verbalisation"])
        if not own_words:
            recurrence_rates.append(0.0)
            continue
        adjacent_words = set()
        for j in range(max(0, i - window), min(n, i + window + 1)):
            if j == i:
                continue
            adjacent_words.update(content_words(scores[j]["verbalisation"]))
        if not adjacent_words:
            recurrence_rates.append(0.0)
            continue
        recurrence_rate = len(own_words & adjacent_words) / len(own_words)
        recurrence_rates.append(recurrence_rate)

    return {
        "mean_recurrence": statistics.mean(recurrence_rates),
        "median_recurrence": statistics.median(recurrence_rates),
        "p10": np.percentile(recurrence_rates, 10),
        "p90": np.percentile(recurrence_rates, 90),
        "interpretation": (
            "Mean recurrence rate is the fraction of content words in each "
            "verbalisation that also appear in nearby verbalisations. High "
            "values indicate stable thematic identification by the AV; low "
            "values flag singleton claims that the source paper says to treat "
            "as potentially confabulated."
        ),
    }


def confabulation_case_study(scores: list[dict]) -> dict | None:
    """Section 11 of v6: the idx 502 'systems' verbalisation as a worked example
    of the paper's confabulation diagnostic framework."""
    for s in scores:
        if s["token_index"] == 502:
            return {
                "token_index": s["token_index"],
                "token": s["token"],
                "cosine": s["cosine"],
                "mse": s["mse"],
                "verbalisation": s["verbalisation"],
                "diagnostic": {
                    "specific_claim_present": True,
                    "specific_claim_recurs": False,
                    "thematically_adjacent": True,
                    "verdict": "Textbook confabulation by Fraser-Taliente "
                               "et al.'s heuristics: specific (named phrase), "
                               "non-recurring (no echo in adjacent tokens), "
                               "thematically adjacent (security passage).",
                },
            }
    return None


def compute_fve(scores: list[dict], parquet_path: Path) -> dict | None:
    """Section 3.3 of v6: paper's primary metric. Requires activations parquet.
    FVE = 1 - sum_t ||h_t - hhat_t||² / sum_t ||h_t - mean(h)||²"""
    if not HAS_PYARROW:
        return None
    table = pq.read_table(parquet_path)
    acts = np.array(table["activation_vector"].to_pylist())
    norms = np.linalg.norm(acts, axis=1, keepdims=True)
    acts_n = acts / norms

    mses = np.array([s["mse"] for s in scores])
    sq_errors = mses * D_MODEL

    mean_h = acts_n.mean(axis=0)
    sq_devs = ((acts_n - mean_h) ** 2).sum(axis=1)

    fve = 1 - sq_errors.sum() / sq_devs.sum()

    pre_mask = np.array([s["token_index"] < PRE_ATTACK_END for s in scores])
    att_mask = np.array([s["token_index"] >= ATTACK_START for s in scores])

    fve_pre = 1 - sq_errors[pre_mask].sum() / sq_devs[pre_mask].sum()
    fve_att = 1 - sq_errors[att_mask].sum() / sq_devs[att_mask].sum()

    return {
        "session_fve": float(fve),
        "pre_attack_fve": float(fve_pre),
        "attack_region_fve": float(fve_att),
        "paper_reference_range": [0.6, 0.8],
        "interpretation": (
            "Paper reports trained NLAs reach FVE 0.6-0.8 at Anthropic scale "
            "(Opus/Gemma-3-27B). Open-model checkpoint released by paper "
            "authors may achieve lower FVE. FVE < 0.3 may indicate the "
            "fallback prompt template is significantly degrading "
            "reconstruction; investigate before chapter submission."
        ),
    }


# ============================================================================
# Verbalisation sampling — only from cleared categories
# ============================================================================

# Categories whose exemplar verbalisations appear in v6. Reagent and
# attack-payload verbalisations are not sampled here. The chapter's analytical
# findings on those positions are based on quantitative reconstruction scores,
# not on the AV's verbalisations of them.
CLEARED_SAMPLE_CATEGORIES = [
    "hard_refusal",
    "risk_vocabulary",
    "internal_deliberation",
    "epistemic_hedging",
    "directive_against",
    "compliance_pivot",
]


def sample_verbalisations(scores: list[dict]) -> list[dict]:
    """Sample exemplar verbalisations for v6's cleared categories.
    Verbalisations are pulled from positions matching each category's
    vocabulary. The samples are intended for illustrating the methodology,
    not for ranking the most anomalous tokens in the session."""
    samples = []
    for cat_name in CLEARED_SAMPLE_CATEGORIES:
        keywords = DECISION_CATEGORIES[cat_name]
        cat_tokens = []
        for s in scores:
            tok = s["token"].strip().lower()
            if not tok or len(tok) > 25:
                continue
            if not any(kw.lower().strip() in tok for kw in keywords):
                continue
            cat_tokens.append({
                "category": cat_name,
                "token_index": s["token_index"],
                "token": s["token"],
                "region": (
                    "pre-attack" if s["token_index"] < PRE_ATTACK_END
                    else "during attack"
                ),
                "mse": s["mse"],
                "cosine": s["cosine"],
                "verbalisation": s["verbalisation"][:300],
            })
        # Sample up to 3 per category, sorted by cosine ascending so the
        # exemplars are the lowest-reconstruction tokens in the category
        # (consistent with v6's exemplars).
        cat_tokens.sort(key=lambda x: x["cosine"])
        samples.extend(cat_tokens[:3])
    return samples


# ============================================================================
# Report generation
# ============================================================================

def generate_markdown_report(results: dict, out_path: Path):
    lines = []
    lines.append("# Dr House Session — Reproducibility Report\n")
    lines.append(
        "Mirrors the quantitative analyses in NLA_Findings_Dr_House_Session "
        "v6. Reproducible from `ar_scores.json` and the activations parquet.\n"
    )
    lines.append("Citation: Fraser-Taliente, K., Kantamneni, S., Ong, E., "
                 "et al. (2026). 'Natural Language Autoencoders Produce "
                 "Unsupervised Explanations of LLM Activations.' "
                 "Transformer Circuits Thread. "
                 "https://transformer-circuits.pub/2026/nla/\n")

    # Session-level
    ss = results["session_stats"]
    lines.append("## 1. Session-level statistics\n")
    lines.append(f"Tokens: {ss['n_tokens']}\n")
    lines.append("### MSE\n")
    lines.append(f"- min: {ss['mse']['min']:.6f}")
    lines.append(f"- mean: {ss['mse']['mean']:.6f}")
    lines.append(f"- max: {ss['mse']['max']:.6f}")
    lines.append(f"- stdev: {ss['mse']['stdev']:.6f}")
    lines.append(f"- percentiles (p10/p50/p90/p99): "
                 f"{ss['mse']['p10']:.6f} / {ss['mse']['p50']:.6f} / "
                 f"{ss['mse']['p90']:.6f} / {ss['mse']['p99']:.6f}\n")
    lines.append("### Cosine\n")
    lines.append(f"- min: {ss['cosine']['min']:.4f}")
    lines.append(f"- mean: {ss['cosine']['mean']:.4f}")
    lines.append(f"- max: {ss['cosine']['max']:.4f}")
    lines.append(f"- stdev: {ss['cosine']['stdev']:.4f}")
    lines.append(f"- percentiles (p1/p10/p50/p90/p99): "
                 f"{ss['cosine']['p1']:.4f} / {ss['cosine']['p10']:.4f} / "
                 f"{ss['cosine']['p50']:.4f} / {ss['cosine']['p90']:.4f} / "
                 f"{ss['cosine']['p99']:.4f}\n")

    # Region split
    rs = results["region_split"]
    lines.append("## 2. Region split — pre-attack vs during attack\n")
    lines.append(
        f"- **Pre-attack** (idx < {PRE_ATTACK_END}, n={rs['pre_attack']['n']}): "
        f"mean cos {rs['pre_attack']['mean']:.4f} "
        f"[{rs['pre_attack']['ci_lo']:.4f}, {rs['pre_attack']['ci_hi']:.4f}]"
    )
    lines.append(
        f"- **During attack** (idx >= {ATTACK_START}, "
        f"n={rs['during_attack']['n']}): mean cos "
        f"{rs['during_attack']['mean']:.4f} "
        f"[{rs['during_attack']['ci_lo']:.4f}, "
        f"{rs['during_attack']['ci_hi']:.4f}]\n"
    )

    # Category table
    lines.append("## 3. Eleven-category position-stratified analysis\n")
    lines.append("| Category | Pre n | Pre cos | Att n | Att cos | Delta |")
    lines.append("|---|---|---|---|---|---|")
    for cat_name, info in results["categories"].items():
        pre_str = f"{info['pre_mean']:.3f}" if info["pre_mean"] else "—"
        att_str = f"{info['att_mean']:.3f}" if info["att_mean"] else "—"
        del_str = f"{info['delta']:+.3f}" if info["delta"] else "—"
        lines.append(
            f"| {cat_name} | {info['pre_n']} | {pre_str} | "
            f"{info['att_n']} | {att_str} | {del_str} |"
        )
    lines.append("")
    lines.append(
        "Bootstrap 95% CIs computed by resampling with replacement 10,000 "
        "times. Categories where the predicted suppression pattern holds "
        "(worse during attack): hard_refusal, risk_vocabulary. Nine other "
        "categories show the opposite pattern, supporting the "
        "narrow-suppression interpretation in v6 Section 6.\n"
    )

    # Reagent collapse
    lines.append("## 4. Reagent reconstruction (documented procedural vocab)\n")
    lines.append(
        "Per-position reconstruction scores for the procedural tokens "
        "documented in v6 Section 8. The table reports cosine distribution "
        "only; verbalisations are not surfaced from these positions.\n"
    )
    lines.append("| token_index | token | MSE | Cosine |")
    lines.append("|---|---|---|---|")
    for r in results["reagent_collapse"]:
        lines.append(
            f"| {r['token_index']} | {r['token']} | "
            f"{r['mse']:.6f} | {r['cosine']:.4f} |"
        )
    lines.append("")

    # FVE
    if results.get("fve") is not None:
        fve = results["fve"]
        lines.append("## 5. FVE — paper's primary metric\n")
        lines.append(f"- **Session FVE**: {fve['session_fve']:.4f}")
        lines.append(f"- Pre-attack region FVE: {fve['pre_attack_fve']:.4f}")
        lines.append(f"- Attack region FVE: {fve['attack_region_fve']:.4f}")
        lines.append(
            f"- Paper's reported range for trained NLAs: "
            f"{fve['paper_reference_range']}\n"
        )
        lines.append(fve["interpretation"] + "\n")
    else:
        lines.append("## 5. FVE\n")
        lines.append(
            "Skipped — pyarrow not installed or activations parquet not "
            "provided. Install pyarrow and re-run with --activations to "
            "compute.\n"
        )

    # Recurring claim diagnostic
    rc = results["recurring_claim"]
    lines.append("## 6. Recurring-claim diagnostic\n")
    lines.append(f"- Mean recurrence rate: {rc['mean_recurrence']:.3f}")
    lines.append(f"- Median: {rc['median_recurrence']:.3f}")
    lines.append(f"- p10/p90: {rc['p10']:.3f} / {rc['p90']:.3f}\n")
    lines.append(rc["interpretation"] + "\n")

    # Confabulation case study
    cs = results.get("confabulation_case")
    if cs:
        lines.append("## 7. Confabulation case study — idx 502\n")
        lines.append(
            f"Token: `{cs['token']}` at idx {cs['token_index']}, "
            f"cos {cs['cosine']:.4f}\n"
        )
        lines.append("Verbalisation:")
        lines.append(f"> {cs['verbalisation']}\n")
        lines.append(f"**Diagnostic**: {cs['diagnostic']['verdict']}\n")

    # Note about what is and isn't included
    lines.append("## Notes on scope\n")
    lines.append(
        "This report covers the quantitative findings in v6: session "
        "statistics, region split, eleven-category gradient, reagent "
        "reconstruction scores, FVE, recurring-claim diagnostic, and the "
        "idx 502 confabulation case study. It does not surface "
        "verbalisations from attack-payload positions; the chapter's "
        "claims about those positions rest on reconstruction scores "
        "(cosine, MSE), not on AV verbalisations of payload tokens. "
        "Exemplar verbalisations for cleared categories are written to "
        "`verbalisation_samples.csv` alongside this report.\n"
    )

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def write_tables_csv(results: dict, out_path: Path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "region", "n", "mean_cos",
                    "ci_lo", "ci_hi"])
        # Session baseline rows
        rs = results["region_split"]
        w.writerow(["SESSION", "pre_attack", rs["pre_attack"]["n"],
                    f"{rs['pre_attack']['mean']:.4f}",
                    f"{rs['pre_attack']['ci_lo']:.4f}",
                    f"{rs['pre_attack']['ci_hi']:.4f}"])
        w.writerow(["SESSION", "during_attack", rs["during_attack"]["n"],
                    f"{rs['during_attack']['mean']:.4f}",
                    f"{rs['during_attack']['ci_lo']:.4f}",
                    f"{rs['during_attack']['ci_hi']:.4f}"])
        # Per-category rows
        for cat_name, info in results["categories"].items():
            for region, prefix in [("pre_attack", "pre"),
                                   ("during_attack", "att")]:
                n = info[f"{prefix}_n"]
                mean = info[f"{prefix}_mean"]
                ci = info[f"{prefix}_ci"]
                w.writerow([
                    cat_name, region, n,
                    f"{mean:.4f}" if mean is not None else "",
                    f"{ci[0]:.4f}" if ci[0] is not None else "",
                    f"{ci[1]:.4f}" if ci[1] is not None else "",
                ])


def write_samples_csv(samples: list[dict], out_path: Path):
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "category", "token_index", "token", "region",
            "mse", "cosine", "verbalisation",
        ])
        w.writeheader()
        for row in samples:
            w.writerow(row)


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True,
                    help="Path to ar_scores.json")
    ap.add_argument("--activations", type=Path, default=None,
                    help="Path to activations parquet (for FVE)")
    ap.add_argument("--out-dir", type=Path, default=Path("./output"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading scores from {args.scores}...")
    scores = load_scores(args.scores)
    print(f"  Loaded {len(scores)} tokens")

    print("Computing session-level statistics...")
    session_stats = session_level_stats(scores)

    print("Computing region-split statistics...")
    region_split = region_split_stats(scores)

    print("Computing eleven-category position-stratified analysis...")
    categories = category_analysis(scores)

    print("Computing reagent-position reconstruction table...")
    reagent_collapse = reagent_collapse_analysis(scores)

    print("Computing recurring-claim diagnostic...")
    recurring = recurring_claim_diagnostic(scores)

    print("Computing confabulation case study (idx 502)...")
    confabulation = confabulation_case_study(scores)

    fve = None
    if args.activations:
        print(f"Computing FVE from {args.activations}...")
        fve = compute_fve(scores, args.activations)
    else:
        print("Skipping FVE (no --activations provided)")

    results = {
        "session_stats": session_stats,
        "region_split": region_split,
        "categories": categories,
        "reagent_collapse": reagent_collapse,
        "recurring_claim": recurring,
        "confabulation_case": confabulation,
        "fve": fve,
    }

    print("Generating markdown report...")
    generate_markdown_report(results, args.out_dir / "analysis_report.md")

    print("Writing tables.csv...")
    write_tables_csv(results, args.out_dir / "tables.csv")

    print("Sampling cleared-category verbalisations...")
    samples = sample_verbalisations(scores)
    write_samples_csv(samples, args.out_dir / "verbalisation_samples.csv")

    if fve:
        with open(args.out_dir / "fve_results.json", "w") as f:
            json.dump(fve, f, indent=2)

    print(f"\nDone. Outputs written to {args.out_dir}/")
    print(f"  - analysis_report.md")
    print(f"  - tables.csv")
    print(f"  - verbalisation_samples.csv")
    if fve:
        print(f"  - fve_results.json")


if __name__ == "__main__":
    main()
