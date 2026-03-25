"""eval_bootstrap.py — bootstrap confidence intervals for all retrieval metrics

Reads the per-question results stored in the JSON output of eval_comprehensive.py
and resamples at the question level (with replacement) to produce 95% confidence intervals


python3 eval_bootstrap.py \\
    --json ablation_results.json \\
    --n_bootstrap 10000 \\
    --confidence 0.95 \\
    --out_csv bootstrap_ci.csv \\
    --show_ks 1 3 5 10

"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np



# Data loading

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ranks_to_metrics(
    dense_rank_span: int,
    dense_rank_anchor: int,
    ce_rank_span: Optional[int],
    ce_rank_anchor: Optional[int],
    k_max: int,
) -> Dict[str, Any]:
    """Convert per-question ranks to per-question metric """

    def acc(rank: int, k: int) -> int:
        return 1 if rank <= k else 0

    def mrr(rank: int) -> float:
        return 1.0 / rank if rank <= k_max else 0.0

    def ndcg(rank: int) -> float:
        return 1.0 / math.log2(rank + 1) if rank <= k_max else 0.0

    rec: Dict[str, Any] = {}
    for k in range(1, k_max + 1):
        rec[f"acc_span@{k}"] = acc(dense_rank_span,   k)
        rec[f"acc_anchor@{k}"] = acc(dense_rank_anchor, k)
    rec["mrr_span"] = mrr(dense_rank_span)
    rec["mrr_anchor"] = mrr(dense_rank_anchor)
    rec["ndcg_span"] = ndcg(dense_rank_span)

    if ce_rank_span is not None:
        for k in range(1, k_max + 1):
            rec[f"ce_acc_span@{k}"] = acc(ce_rank_span,   k)
            rec[f"ce_acc_anchor@{k}"] = acc(ce_rank_anchor, k)
        rec["ce_mrr_span"] = mrr(ce_rank_span)
        rec["ce_mrr_anchor"] = mrr(ce_rank_anchor)
        rec["ce_ndcg_span"] = ndcg(ce_rank_span)

    return rec


def extract_question_records(
    json_data: Dict[str, Any],
    k_max: int,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """extract per-question metric records from JSON"""

    result: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for model_entry in json_data.get("models", []):
        name = model_entry["name"]
        pooling = model_entry.get("pooling", "lmk")
        key = f"{name}|{pooling}"
        result[key] = {"AGGREGATE": []}

        for paper in model_entry.get("papers", []):
            pid = str(paper["paper_id"])
            per_q: List[Dict[str, Any]] = []

            for qr in paper.get("per_question", []):
                #k_max from config (fall back to 20 if not stored)
                cfg_kmax = json_data.get("config", {}).get("k_max", k_max)
                rec = _ranks_to_metrics(
                    dense_rank_span=int(qr.get("dense_rank_span", cfg_kmax + 1)),
                    dense_rank_anchor=int(qr.get("dense_rank_anchor", cfg_kmax + 1)),
                    ce_rank_span=(
                        int(qr["ce_rank_span"])
                        if qr.get("ce_rank_span") is not None else None
                    ),
                    ce_rank_anchor=(
                        int(qr["ce_rank_anchor"])
                        if qr.get("ce_rank_anchor") is not None else None
                    ),
                    k_max=cfg_kmax,
                )
                per_q.append(rec)

            result[key][pid] = per_q
            result[key]["AGGREGATE"].extend(per_q)

    return result


#-------------------
#-----Bootstrap-----
#-------------------

def _mean_over_records(records: List[Dict[str, Any]], metric: str) -> float:
    vals = [r[metric] for r in records if metric in r]
    return float(np.mean(vals)) if vals else np.nan


def bootstrap_ci(
    records: List[Dict[str, Any]],
    metrics: List[str],
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Tuple[float, float, float]]:
    """Bootstrap confidence intervals for a list of per-question metrics

    Returns:
        {metric: (mean, ci_low, ci_high)}
    """
    if rng is None:
        rng = np.random.default_rng(seed=0)

    Q = len(records)
    if Q == 0:
        return {m: (np.nan, np.nan, np.nan) for m in metrics}

    #build a 2-D array [Q, n_metrics] for fast resampling
    valid_metrics = [m for m in metrics if m in records[0]]
    arr = np.array([[r[m] for m in valid_metrics] for r in records], dtype=np.float32)

    #Bootstrap: resample Q questions with replacement, compute mean per sample
    boot_means = np.empty((n_bootstrap, len(valid_metrics)), dtype=np.float32)
    for i in range(n_bootstrap):
        idx = rng.integers(0, Q, size=Q)
        boot_means[i] = arr[idx].mean(axis=0)

    alpha = 1.0 - confidence
    lo = np.percentile(boot_means, 100 * alpha / 2,     axis=0)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2), axis=0)

    result: Dict[str, Tuple[float, float, float]] = {}
    for j, m in enumerate(valid_metrics):
        result[m] = (float(arr[:, j].mean()), float(lo[j]), float(hi[j]))


    for m in metrics:
        if m not in result:
            result[m] = (np.nan, np.nan, np.nan)

    return result


#--------------------------------
#-------Output helpers-----------
#--------------------------------

def _ci_str(mean: float, lo: float, hi: float) -> str:
    if any(math.isnan(v) for v in [mean, lo, hi]):
        return "  —  "
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def print_ci_table(
    all_ci: List[Dict[str, Any]],
    show_ks: Tuple[int, ...] = (1, 3, 5, 10),
    paper_filter: str = "AGGREGATE",
    ranking_filter: str = "dense",
) -> None:
    rows = [
        r for r in all_ci
        if str(r["paper_scope"]).upper() == paper_filter.upper()
        and r["ranking"] == ranking_filter
    ]
    if not rows:
        print(f"  [no rows for paper_scope={paper_filter}, ranking={ranking_filter}]")
        return

    ks = [k for k in show_ks if f"acc_span@{k}_mean" in rows[0]]
    header_k = "  ".join(f"Acc_s@{k}" for k in ks)

    print(f"\n{'─'*90}")
    print(f"  Bootstrap 95% CIs — {paper_filter} — {ranking_filter.upper()}")
    print(f"  {'Model':<25} {header_k}  MRR_span  NDCG_span")
    print(f"{'─'*90}")

    for r in rows:
        k_parts = [
            _ci_str(r[f"acc_span@{k}_mean"], r[f"acc_span@{k}_ci_low"], r[f"acc_span@{k}_ci_hi"])
            for k in ks
        ]
        mrr  = _ci_str(r["mrr_span_mean"],  r["mrr_span_ci_low"],  r["mrr_span_ci_hi"])
        ndcg = _ci_str(r["ndcg_span_mean"], r["ndcg_span_ci_low"], r["ndcg_span_ci_hi"])
        print(f"  {r['model']:<25} {'  '.join(k_parts)}  {mrr}  {ndcg}")
    print(f"{'─'*90}")


def build_output_rows(
    model_key: str,
    paper_scope: str,
    ranking: str,
    n_q: int,
    n_bootstrap: int,
    ci: Dict[str, Tuple[float, float, float]],
) -> Dict[str, Any]:
    name, pooling = model_key.split("|", 1)
    row: Dict[str, Any] = {
        "model": name,
        "pooling": pooling,
        "ranking": ranking,
        "paper_scope": paper_scope,
        "n_questions": n_q,
        "n_bootstrap": n_bootstrap,
    }
    for metric, (mean, lo, hi) in ci.items():
        row[f"{metric}_mean"]   = round(mean, 6) if not math.isnan(mean) else None
        row[f"{metric}_ci_low"] = round(lo,   6) if not math.isnan(lo)   else None
        row[f"{metric}_ci_hi"]  = round(hi,   6) if not math.isnan(hi)   else None
    return row


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        print("[WARNING] No rows to write.", file=sys.stderr)
        return
    #collect all fieldnames preserving order of first appearance
    seen: Dict[str, None] = {}
    for r in rows:
        seen.update(dict.fromkeys(r.keys()))
    fields = list(seen.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n[CSV]  {path}")


#--------------------------------
#-------Entry point--------------
#--------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--json", required=True,
        help="JSON output from eval_comprehensive.py (--out_json path).",
    )
    ap.add_argument("--n_bootstrap", type=int,   default=10_000)
    ap.add_argument("--confidence",  type=float, default=0.95)
    ap.add_argument("--k_max",       type=int,   default=10)
    ap.add_argument(
        "--show_ks", nargs="+", type=int, default=[1, 3, 5, 10],
        help="K values to show in the printed table.",
    )
    ap.add_argument(
        "--per_paper", action="store_true",
        help="Also compute per-paper CIs (in addition to aggregate).",
    )
    ap.add_argument(
        "--models", nargs="*", default=[],
        help="If set, restrict output to these model names.",
    )
    ap.add_argument(
        "--out_csv", type=str, default="bootstrap_ci.csv",
        help="Output path for CI results CSV.",
    )
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    if not Path(args.json).exists():
        ap.error(
            f"JSON file not found: {args.json}\n"
            "Generate it with:  eval_comprehensive.py ... --out_json ablation_results.json"
        )

    print(f">>> Loading {args.json} ...")
    data = load_json(args.json)
    k_max = data.get("config", {}).get("k_max", args.k_max)

    print(f">>> Extracting per-question records (k_max={k_max}) ...")
    all_records = extract_question_records(data, k_max=k_max)

    rng = np.random.default_rng(seed=args.seed)

    #determine metrics to bootstrap
    dense_metrics = (
        [f"acc_span@{k}" for k in range(1, k_max + 1)]
        + [f"acc_anchor@{k}" for k in range(1, k_max + 1)]
        + ["mrr_span", "mrr_anchor", "ndcg_span"]
    )
    ce_metrics = (
        [f"ce_acc_span@{k}" for k in range(1, k_max + 1)]
        + [f"ce_acc_anchor@{k}" for k in range(1, k_max + 1)]
        + ["ce_mrr_span", "ce_mrr_anchor", "ce_ndcg_span"]
    )

    output_rows: List[Dict[str, Any]] = []
    all_ci_flat: List[Dict[str, Any]] = []  #for printing

    for model_key, paper_dict in all_records.items():
        name = model_key.split("|")[0]
        if args.models and name not in args.models:
            continue

        scopes = ["AGGREGATE"] + (
            [k for k in paper_dict if k != "AGGREGATE"] if args.per_paper else []
        )

        for scope in scopes:
            records = paper_dict.get(scope, [])
            if not records:
                continue
            print(f"  {name:<30} scope={scope:<12} n_q={len(records)}", end="", flush=True)

            #dense bootstrap
            ci_dense = bootstrap_ci(records, dense_metrics, args.n_bootstrap, args.confidence, rng)
            row_d = build_output_rows(model_key, scope, "dense", len(records), args.n_bootstrap, ci_dense)
            output_rows.append(row_d)

            #check whether CE data exists
            has_ce = any(ce_metrics[0] in r for r in records)
            if has_ce:
                ci_ce = bootstrap_ci(records, ce_metrics, args.n_bootstrap, args.confidence, rng)
                #Remap ce_ prefixed keys to plain names for uniform output row structure
                ci_ce_remapped = {
                    k.replace("ce_", ""): v for k, v in ci_ce.items()
                }
                row_ce = build_output_rows(model_key, scope, "ce", len(records), args.n_bootstrap, ci_ce_remapped)
                output_rows.append(row_ce)
                print(f"  [dense+CE]", flush=True)
            else:
                print(f"  [dense only]", flush=True)

            all_ci_flat.append(row_d)

    #print summary table.
    print_ci_table(all_ci_flat, show_ks=tuple(args.show_ks), ranking_filter="dense")

    
    write_csv(output_rows, args.out_csv)
    print(f"\n>>> Bootstrap complete.  {args.n_bootstrap} iterations, {args.confidence*100:.0f}% CIs.")
    print("    Feed bootstrap_ci.csv to plot_results_v3.py to add error bars to all plots.")


if __name__ == "__main__":
    main()
