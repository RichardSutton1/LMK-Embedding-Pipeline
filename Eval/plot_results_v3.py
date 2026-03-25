
"""plot_results_v3.py — ablation plots 


Generates 10 plots:
  1. acc_at_k.png (dense + CE panels)
  2. negative_strategy.png 
  3. layers_unfrozen.png 
  4. acc_vs_epoch.png
  5. window_size.png   
  6. pooling_comparison.png
  7. mrr_comparison.png 
  8. temperature.png 
  9. recall_at_k.png 
 10. per_document.png (per-paper Acc@1)


python3 plot_results_v3.py --csv ablation_results.csv
python3 plot_results_v3.py \\
    --csv ablation_results.csv \\
    --bootstrap_csv bootstrap_ci.csv \\
    --out_dir plots_v3/
"""

from __future__ import annotations
import argparse
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
import platform

if platform.system() != "Darwin":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


OKI = dict(
    blue  = "#0072B2",
    vermillion = "#D55E00",
    green = "#009E73",
    amber      = "#E69F00",
    sky        = "#56B4E9",
    pink       = "#CC79A7",
    yellow     = "#F0E442",
    black      = "#000000",
)

PALETTE: Dict[str, Any] = {
    "lmk_ft":    OKI["blue"],
    "base_lmk":  OKI["vermillion"],
    "base_cls":  OKI["green"],
    "base_mean": OKI["amber"],
    # negative strategy
    "a1": OKI["blue"],
    "a2": OKI["sky"],
    "a3": OKI["green"],
    "a4": OKI["vermillion"],
    # layers unfrozen
    "b1": OKI["amber"],
    "b2": OKI["sky"],
    "b3": OKI["green"],
    "b4": OKI["blue"],
    "dense": OKI["blue"],
    "ce":    OKI["vermillion"],
    "generic": [
        OKI["blue"], OKI["vermillion"], OKI["green"], OKI["amber"],
        OKI["sky"],  OKI["pink"],       "#332288",    "#117733",
    ],
}

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize":11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.linewidth":0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.size":  4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.8,
    "ytick.major.width":0.8,
    "legend.fontsize":  8.5,
    "legend.framealpha": 0.85,
    "legend.edgecolor":"0.8",
    "legend.handlelength": 1.8,
    "grid.alpha": 0.20,
    "grid.linestyle":":",
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "patch.linewidth": 0.5,
    "errorbar.capsize": 3,
})


# ---- helpers ----

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def aggregate_rows(df: pd.DataFrame, dense_only: bool = True) -> pd.DataFrame:
    """Return AGGREGATE rows, 
    (optional dense-only)
    """
    if "paper_id" not in df.columns:
        return df
    agg = df[df["paper_id"].astype(str).str.lower() == "aggregate"].copy()
    if agg.empty:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        agg = df.groupby("model")[numeric].mean().reset_index()
    if dense_only and "reranked" in agg.columns:
        dense = agg[agg["reranked"].astype(str).str.lower().isin(["false", "0", "no"])]
        if not dense.empty:
            return dense
    return agg


def paper_rows(df: pd.DataFrame, dense_only: bool = True) -> pd.DataFrame:
    """return per-paper rows """
    if "paper_id" not in df.columns:
        return df
    per_paper = df[df["paper_id"].astype(str).str.lower() != "aggregate"].copy()
    if dense_only and "reranked" in per_paper.columns:
        dense = per_paper[
            per_paper["reranked"].astype(str).str.lower().isin(["false", "0", "no"])
        ]
        if not dense.empty:
            return dense
    return per_paper



def ce_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "reranked" not in df.columns:
        return pd.DataFrame()
    agg = (
        df[df["paper_id"].astype(str).str.lower() == "aggregate"].copy()
        if "paper_id" in df.columns
        else df.copy()
    )
    return agg[agg["reranked"].astype(str).str.lower().isin(["true", "1", "yes"])]




def acc_span_cols(df: pd.DataFrame) -> List[str]:
    return sorted(
        [c for c in df.columns if c.startswith("acc_span@")],
        key=lambda x: int(x.split("@")[1]),
    )


def acc_anchor_cols(df: pd.DataFrame) -> List[str]:
    return sorted(
        [c for c in df.columns if c.startswith("acc_anchor@")],
        key=lambda x: int(x.split("@")[1]),
    )



def get_val(df: pd.DataFrame, model: str, col: str, default: float = np.nan) -> float:
    rows = df[df["model"] == model]
    if rows.empty or col not in df.columns:
        return default
    return float(rows[col].values[0])



def save_fig(fig: Any, path: str) -> None:
    fig.tight_layout(pad=0.4)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  ✓  {path}")


# ---- bootstrap CI helpers ---

def load_bootstrap(path: str) -> pd.DataFrame:
    """Load bootstrap_ci.csv"""
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    df = load_csv(path)
    if "paper_scope" in df.columns:
        df = df[df["paper_scope"].astype(str).str.upper() == "AGGREGATE"].copy()
    if "ranking" in df.columns:
        df = df[df["ranking"].astype(str).str.lower() == "dense"].copy()
    return df



def get_ci(
    bdf: pd.DataFrame,
    model: str,
    metric: str,
) -> Tuple[float, float, float]:
    """Returns (mean, ci_low, ci_hi) from bootstrap"""

    if bdf.empty:
        return np.nan, np.nan, np.nan
    row = bdf[bdf["model"] == model]

    if row.empty:
        return np.nan, np.nan, np.nan

    mean = float(row.get(f"{metric}_mean",   pd.Series([np.nan])).values[0])
    lo   = float(row.get(f"{metric}_ci_low", pd.Series([np.nan])).values[0])
    hi   = float(row.get(f"{metric}_ci_hi",  pd.Series([np.nan])).values[0])
    return mean, lo, hi




def _ci_yerr(
    bdf: pd.DataFrame,
    model_ids: List[str],
    metric: str,
    vals: List[float],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """builds error-bar arrays from bootstrap CI"""
    if bdf.empty:
        return None

    ci_list = [get_ci(bdf, m, metric) for m in model_ids]
    lo_arr  = np.array([c[1] for c in ci_list])
    hi_arr  = np.array([c[2] for c in ci_list])

    if np.all(np.isnan(lo_arr)):
        return None
    v = np.array(vals)
    return ( np.where(np.isnan(lo_arr), 0.0, v - lo_arr),np.where(np.isnan(hi_arr), 0.0, hi_arr - v),)


# --- Acc@K curves ---

def plot_acc_at_k(
    df: pd.DataFrame,
    bdf: pd.DataFrame,
    out_dir: str,
    use_span: bool = True,
) -> None:
    """Line plot of Acc@K for every model - dense + CE panels"""
    col_fn     = acc_span_cols if use_span else acc_anchor_cols
    span_label = "Span" if use_span else "Anchor"
    dense  = aggregate_rows(df, dense_only=True)
    ce     = ce_rows(df)
    k_cols = col_fn(dense)
    if len(k_cols) < 2:
        print("[skip] acc@k -- not enough k columns")
        return
    ks     = [int(c.split("@")[1]) for c in k_cols]
    models = dense["model"].unique().tolist()
    panels = [(dense, "Dense-Only"), (ce, "CE Reranked")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (rows, title) in zip(axes, panels):
        if rows.empty:
            ax.set_title(f"Acc@K ({span_label}) — {title}\n(no data)")
            ax.axis("off")
            continue
        for idx, model in enumerate(models):
            mrow = rows[rows["model"] == model]
            if mrow.empty:
                continue
            vals   = [float(mrow[c].values[0]) if c in mrow.columns else np.nan for c in k_cols]
            color  = PALETTE["generic"][idx % len(PALETTE["generic"])]
            marker = MARKERS[idx % len(MARKERS)]
            ax.plot(ks, vals, marker=marker, color=color, label=model, zorder=3)
            if title == "Dense-Only" and not bdf.empty:
                lo_v = [get_ci(bdf, model, c)[1] for c in k_cols]
                hi_v = [get_ci(bdf, model, c)[2] for c in k_cols]
                if not all(np.isnan(lo_v)):
                    ax.fill_between(ks, lo_v, hi_v, color=color, alpha=0.12)
        ax.set_xlabel("K")
        ax.set_ylabel(f"Acc@K ({span_label})")
        ax.set_title(f"Acc@K ({span_label}) — {title}")
        ax.set_xticks(ks)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.grid(True)
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    fig.legend(handles, labels, ncol=min(4, len(handles)), fontsize=7.5,
               loc="lower center", bbox_to_anchor=(0.5, -0.18),
               framealpha=0.85, edgecolor="0.8")
    save_fig(fig, os.path.join(out_dir, "acc_at_k.png"))



# ---2: negative strategy ---

def plot_negative_strategy(
    df: pd.DataFrame,
    bdf: pd.DataFrame,
    out_dir: str,
    args: Any,
) -> None:
    """effect of negative mining strategy on Acc@K."""
    dense = aggregate_rows(df, dense_only=True)
    strategy_models = {
        "Baseline\n(no FT)": args.base_lmk,
        "In-batch\n(A1)":args.a1_model,
        "Mined\n(A2)": args.a2_model,
        "Both\n(A3)": args.a3_model,
        "Random\n(A4)": args.a4_model,
    }
    present = {k: v for k, v in strategy_models.items() if v and v in dense["model"].values}
    if len(present) < 2:
        print(f"  [skip] negative strategy -- need >=2 models (found: {list(present)})")
        return
    metrics = (
        [c for c in ["acc_span@1", "acc_span@3", "acc_span@5", "acc_span@10"] if c in dense.columns]
        or [c for c in ["acc_anchor@1", "acc_anchor@3", "acc_anchor@5"] if c in dense.columns]
    )
    if not metrics:
        print("[skip] negative strategy -- no acc collumns")
        return

    names = list(present.keys())
    model_ids = list(present.values())
    x = np.arange(len(names))
    width = 0.8 / len(metrics)
    colors = [OKI["blue"], OKI["sky"], OKI["green"], OKI["amber"]][: len(metrics)]

    fig, ax = plt.subplots(figsize=(max(9, len(names) * 1.9), 5.5))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [get_val(dense, m, metric, 0.0) for m in model_ids]
        offset = (i - len(metrics) / 2 + 0.5) * width
        yerr = _ci_yerr(bdf, model_ids, metric, vals)
        label = metric.replace("acc_span@", "Acc@").replace("acc_anchor@", "Acc@")
        bars = ax.bar(
            x + offset, vals, width * 0.9, label=label, color=color,
            yerr=yerr, error_kw=dict(elinewidth=0.8, ecolor="0.3", capsize=2),
        )
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_xlabel("Negative Mining Strategy")
    ax.set_ylabel("Accuracy (Dense-Only)")
    ax.set_title("Effect of Negative Mining Strategy on Retrieval Accuracy")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=8.5)
    ax.grid(axis="y")
    save_fig(fig, os.path.join(out_dir, "negative_strategy.png"))





# --- 3: layers unfrozen ---

def plot_layers_unfrozen(
    df: pd.DataFrame,
    bdf: pd.DataFrame,
    out_dir: str,
    args: Any,
) -> None:
    """layers unfrozen vs acc"""
    dense = aggregate_rows(df, dense_only=True)
    layer_map = [
        ("Baseline\n(0 layers)", args.base_lmk),
        ("LMK token\nonly", args.b1_model),
        ("Last 2\nlayers", args.b2_model),
        ("Last 4\nlayers", args.b3_model),
        ("Full\nfinetune", args.b4_model),
    ]
    present = [(lbl, mid) for lbl, mid in layer_map if mid and mid in dense["model"].values]
    if len(present) < 2:
        print("[skip] layers unfrozen - need >=2 models")
        return
    labels, model_ids = zip(*present)
    metrics = (
        [c for c in ["acc_span@1", "acc_span@5", "acc_span@10"] if c in dense.columns]
        or [c for c in ["acc_anchor@1", "acc_anchor@5", "acc_anchor@10"] if c in dense.columns]
    )
    if not metrics:
        print("[skip] layers unfrozen - no acc columns")
        return

    x = np.arange(len(labels))
    colors = [OKI["blue"], OKI["green"], OKI["amber"]][: len(metrics)]
    fig, ax = plt.subplots(figsize=(10, 5))
    for metric, color, mkr in zip(metrics, colors, MARKERS):
        vals = [get_val(dense, m, metric) for m in model_ids]
        label = metric.replace("acc_span@", "Acc@").replace("acc_anchor@", "Acc@")
        ax.plot(x, vals, marker=mkr, color=color, label=label, zorder=3)
        if not bdf.empty:
            lo_v = [get_ci(bdf, m, metric)[1] for m in model_ids]
            hi_v = [get_ci(bdf, m, metric)[2] for m in model_ids]
            if not all(np.isnan(lo_v)):
                ax.fill_between(x, lo_v, hi_v, color=color, alpha=0.12)
        for xi, v in enumerate(vals):
            if not np.isnan(v):
                ax.annotate(
                    f"{v:.2f}", (xi, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Training Configuration (Layers Unfrozen)")
    ax.set_ylabel("Accuracy (Dense-Only)")
    ax.set_title("Effect of Unfreezing Layers on Retrieval Accuracy")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=8.5)
    ax.grid(True)
    save_fig(fig, os.path.join(out_dir, "layers_unfrozen.png"))



# --- 4: acc vs epoch ---

def plot_acc_vs_epoch(
    epoch_csv: str,
    main_df: pd.DataFrame,
    bdf: pd.DataFrame,
    out_dir: str,
    args: Any,
) -> None:
    """accuracy vs training epoch"""
    if not epoch_csv or not os.path.exists(epoch_csv):
        print("  [skip] acc vs epoch -- --epoch_csv not provided or not found")
        return
    df = load_csv(epoch_csv)
    dense = aggregate_rows(df, dense_only=True)
    if "epoch" not in dense.columns:
        dense = dense.copy()
        dense["epoch"] = dense["model"].str.extract(r"_ep(\d+)$").astype(float)
        dense["run"] = dense["model"].str.replace(r"_ep\d+$", "", regex=True)
    else:
        dense["run"] = dense.get("run", dense["model"])
    dense = dense.dropna(subset=["epoch"])
    if dense.empty:
        print("  [skip] acc vs epoch -- could not parse epoch numbers")
        return
    metrics = (
        [c for c in ["acc_span@1", "acc_span@5", "acc_span@10"] if c in dense.columns]
        or [c for c in ["acc_anchor@1", "acc_anchor@5", "acc_anchor@10"] if c in dense.columns]
    )
    if not metrics:
        print("  [skip] acc vs epoch -- no acc columns")
        return

    base_dense = aggregate_rows(main_df, dense_only=True)
    colors  = [OKI["blue"], OKI["vermillion"], OKI["green"]]
    fig, axes  = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric, color in zip(axes, metrics, colors):
        label = metric.replace("acc_span@", "Acc@").replace("acc_anchor@", "Acc@")
        for run, grp in dense.groupby("run"):
            grp= grp.sort_values("epoch")
            ax.plot(grp["epoch"], grp[metric], marker="o", color=color, label=run)
        bval = get_val(base_dense, args.base_lmk, metric)
        if not np.isnan(bval):
            ax.axhline(
                bval, linestyle="--", color="#888", linewidth=1.2,
                label=f"Baseline ({bval:.2f})",
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs Epoch")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.legend(fontsize=8)
        ax.grid(True)
    save_fig(fig, os.path.join(out_dir, "acc_vs_epoch.png"))



# ----5: window size ----

def plot_window_size(window_csvs: List[str], out_dir: str) -> None:
    """context window size vs accuracy."""
    if not window_csvs:
        print("[skip] window size - no --window_csvs provided")
        return
    records = []
    for path in window_csvs:
        if not os.path.exists(path):
            print(f"[warn] window CSV not found: {path}")
            continue
        df = load_csv(path)
        agg = aggregate_rows(df, dense_only=True)
        if agg.empty:
            continue
        m = re.search(r"w(\d+)", os.path.basename(path))
        w = int(m.group(1)) if m else len(records)
        for _, row in agg.iterrows():
            records.append({"window": w, **row.to_dict()})
    if len(records) < 2:
        print("[skip] window size - need >=2 window CSVs with data")
        return
    wdf = pd.DataFrame(records).sort_values("window")
    metrics = (
        [c for c in ["acc_span@1", "acc_span@5", "acc_span@10"] if c in wdf.columns]
        or [c for c in ["acc_anchor@1", "acc_anchor@5", "acc_anchor@10"] if c in wdf.columns]
    )
    colors = [OKI["blue"], OKI["green"], OKI["amber"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric, color, mkr in zip(metrics, colors, MARKERS):
        label = metric.replace("acc_span@", "Acc@").replace("acc_anchor@", "Acc@")
        for model, grp in wdf.groupby("model"):
            grp = grp.sort_values("window")
            ax.plot(grp["window"], grp[metric], marker=mkr, color=color,
                    label=f"{label} ({model})")
    ax.set_xlabel("Context Window Size (sentences)")
    ax.set_ylabel("Accuracy (Dense-Only)")
    ax.set_title("Effect of Context Window Size on Retrieval Accuracy")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=8)
    ax.grid(True)
    save_fig(fig, os.path.join(out_dir, "window_size.png"))





# --- 6: pooling ---

def plot_pooling_comparison(
    df: pd.DataFrame,
    bdf: pd.DataFrame,
    out_dir: str,
    args: Any,
) -> None:
    """MK vs CLS vs Mean, untrained and fine-tuned."""
    dense = aggregate_rows(df, dense_only=True)
    ce    = ce_rows(df)
    pooling_models = {
        "LMK\n(base)": args.base_lmk,
        "CLS\n(base)": args.base_cls,
        "Mean\n(base)": args.base_mean,
        "LMK\n(fine-tuned)": args.lmk_ft_model,
        "CLS\n(fine-tuned)": getattr(args, "trained_cls",  "CLS_FT16"),
        "Mean\n(fine-tuned)": getattr(args, "trained_mean", "MEAN_FT16"),
    }
    present = {k: v for k, v in pooling_models.items() if v and v in dense["model"].values}
    if len(present) < 2:
        print("  [skip] pooling comparison - need >=2 models")
        return
    names  = list(present.keys())
    model_ids = list(present.values())
    metric = next(
        (c for c in ["acc_span@5", "acc_span@1", "acc_anchor@5"] if c in dense.columns), None
    )
    if metric is None:
        print("  [skip] pooling comparison - no acc columns")
        return
    metric_label = metric.replace("acc_span@", "Acc@").replace("acc_anchor@", "Acc@")

    dense_vals = [get_val(dense, m, metric, 0.0) for m in model_ids]
    ce_vals = (
        [get_val(ce, m, metric, np.nan) for m in model_ids] if not ce.empty
        else [np.nan] * len(model_ids)
    )

    x = np.arange(len(names))
    width = 0.35
    bar_colors_d = [OKI["vermillion"], OKI["green"], OKI["amber"], OKI["blue"], OKI["green"], OKI["amber"]]
    bar_colors_c = [OKI["sky"], OKI["green"], OKI["amber"], OKI["sky"],  OKI["green"], OKI["amber"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    d_yerr = _ci_yerr(bdf, model_ids, metric, dense_vals)
    b1 = ax.bar(
        x - width / 2, dense_vals, width, label="Dense-only",
        color=[bar_colors_d[i % len(bar_colors_d)] for i in range(len(names))],
        yerr=d_yerr, error_kw=dict(elinewidth=0.8, ecolor="0.3", capsize=2),
    )
    b2 = ax.bar(
        x + width / 2, ce_vals, width, label="CE Reranked", alpha=0.8,
        color=[bar_colors_c[i % len(bar_colors_c)] for i in range(len(names))],
    )
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h) and h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.006,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7.5,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_xlabel("Pooling Strategy")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Pooling Strategy Comparison — {metric_label}")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=8.5)
    ax.grid(axis="y")
    save_fig(fig, os.path.join(out_dir, "pooling_comparison.png"))




# ---7 mrr ---

def plot_mrr(df: pd.DataFrame, out_dir: str) -> None:
    """ MRR_span and Acc@1 for all models and dense / CE."""
    dense = aggregate_rows(df, dense_only=True)
    ce = ce_rows(df)
    mrr_col = next((c for c in ["mrr_span", "mrr_anchor", "mrr"] if c in dense.columns), None)
    acc_col = next((c for c in ["acc_span@1", "acc_anchor@1"] if c in dense.columns), None)
    if mrr_col is None or acc_col is None:
        print("[skip] MRR - mrr_span/acc_span@1 not found")
        return
    models = dense["model"].tolist()
    x = np.arange(len(models))
    width  = 0.20

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.5), 5))

    def _bars(vals: List[float], offset: float, color: str, label: str) -> None:
        b = ax.bar(x + offset, vals, width, color=color, label=label, alpha=0.9)
        for bar, v in zip(b, vals):
            if not np.isnan(v) and v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6.5,
                )

    d_mrr = [get_val(dense, m, mrr_col, 0.0) for m in models]
    d_acc = [get_val(dense, m, acc_col, 0.0) for m in models]
    _bars(d_mrr, -width * 1.5, OKI["blue"], "Dense MRR")
    _bars(d_acc, -width * 0.5, OKI["sky"], "Dense Acc@1")
    if not ce.empty:
        c_mrr = [get_val(ce, m, mrr_col, np.nan) for m in models]
        c_acc = [get_val(ce, m, acc_col, np.nan) for m in models]
        _bars(c_mrr, +width * 0.5, OKI["vermillion"], "CE MRR")
        _bars(c_acc, +width * 1.5, OKI["pink"], "CE Acc@1")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("Score")
    ax.set_title("MRR and Acc@1 Comparison — All Models")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=8.5)
    ax.grid(axis="y")
    save_fig(fig, os.path.join(out_dir, "mrr_comparison.png"))



# ---8: temperature sweep ----

def plot_temperature(
    df: pd.DataFrame,
    bdf: pd.DataFrame,
    out_dir: str,
    args: Any,
) -> None:
    """ loss func temp vs acc"""
    dense  = aggregate_rows(df, dense_only=True)
    temp_map = [(0.01, args.d_temp001), (0.05, args.d_temp005), (0.10, args.d_temp01)]
    present  = [(t, m) for t, m in temp_map if m and m in dense["model"].values]
    if len(present) < 2:
        print("[skip] temperature --need >=2 of D_temp001/005/01")
        return
    temps, model_ids = zip(*present)
    metrics = (
        [c for c in ["acc_span@1", "acc_span@5", "acc_span@10"] if c in dense.columns]
        or [c for c in ["acc_anchor@1", "acc_anchor@5", "acc_anchor@10"] if c in dense.columns]
    )
    if not metrics:
        print(" [skip] temperature - no acc columns")
        return

    colors = [OKI["blue"], OKI["green"], OKI["amber"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric, color, mkr in zip(metrics, colors, MARKERS):
        vals = [get_val(dense, m, metric) for m in model_ids]
        label = metric.replace("acc_span@", "Acc@").replace("acc_anchor@", "Acc@")
        ax.plot(list(temps), vals, marker=mkr, color=color, label=label, zorder=3)
        if not bdf.empty:
            lo_v = [get_ci(bdf, m, metric)[1] for m in model_ids]
            hi_v = [get_ci(bdf, m, metric)[2] for m in model_ids]
            if not all(np.isnan(lo_v)):
                ax.fill_between(list(temps), lo_v, hi_v, color=color, alpha=0.12)
        for t, v in zip(temps, vals):
            if not np.isnan(v):
                ax.annotate(
                    f"{v:.2f}", (t, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5,
                )
    ce = ce_rows(df)
    if not ce.empty:
        m0 = metrics[0]
        ce_v = [get_val(ce, m, m0) for m in model_ids]
        lbl = m0.replace("acc_span@", "Acc@").replace("acc_anchor@", "Acc@")
        ax.plot(list(temps), ce_v, marker="s", color=OKI["vermillion"],
                linestyle="--", label=f"{lbl} (CE)")
    ax.set_xlabel("Temperature (τ)")
    ax.set_ylabel("Accuracy (Dense-Only)")
    ax.set_title("Effect of Contrastive Loss Temperature on Retrieval Accuracy")
    ax.set_xticks(list(temps))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=8.5)
    ax.grid(True)
    save_fig(fig, os.path.join(out_dir, "temperature.png"))



# --- 10: per doc breakdown ---

def plot_per_document(
    df: pd.DataFrame,
    bdf: pd.DataFrame,
    out_dir: str,
    args: Any,
) -> None:
    """Acc@1 per paper """
    per_paper = paper_rows(df, dense_only=True)
    if per_paper.empty:
        print(" [skip] per-document -no per-paper rows in CSV")
        return
    acc_col = next(
        (c for c in ["acc_span@1", "acc_anchor@1"] if c in per_paper.columns), None
    )
    if acc_col is None:
        print(" [skip] per-document - no acc@1 column found")
        return

    candidate_models = [
        args.lmk_ft_model, args.base_lmk, args.b4_model, args.a3_model,
    ]
    focus_models = list(dict.fromkeys(
        m for m in candidate_models if m and m in per_paper["model"].values
    ))
    if not focus_models:
        focus_models = per_paper["model"].unique().tolist()[:4]
    if not focus_models:
        print("  [skip] per-document -- no models found")
        return

    raw_ids = per_paper["paper_id"].unique().tolist()
    paper_ids = sorted(
        raw_ids,
        key=lambda pid: (int(str(pid)) if str(pid).isdigit() else float("inf"), str(pid)),
    )
    x = np.arange(len(paper_ids))
    width = 0.8 / len(focus_models)
    colors = [OKI["blue"], OKI["vermillion"], OKI["green"], OKI["amber"]]

    fig, ax = plt.subplots(figsize=(max(12, len(paper_ids) * 1.0), 5))
    for i, (model, color) in enumerate(zip(focus_models, colors)):
        vals = []
        for pid in paper_ids:
            row = per_paper[
                (per_paper["model"] == model) &
                (per_paper["paper_id"].astype(str) == str(pid))
            ]
            vals.append(float(row[acc_col].values[0]) if not row.empty else np.nan)
        offset = (i - len(focus_models) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9, label=model, color=color, alpha=0.88)

    #aggregate performance of best model
    best_model = focus_models[0]
    agg_dense = aggregate_rows(df, dense_only=True)
    agg_val = get_val(agg_dense, best_model, acc_col)
    if not np.isnan(agg_val):
        ax.axhline(
            agg_val, linestyle="--", linewidth=1.2,
            color=OKI["blue"], alpha=0.6,
            label=f"{best_model} aggregate ({agg_val:.2f})",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"P{pid}" for pid in paper_ids], rotation=45, ha="right", fontsize=8.5,
    )
    ax.set_xlabel("Paper / Document")
    ax.set_ylabel(acc_col.replace("acc_span@", "Acc@").replace("acc_anchor@", "Acc@"))
    ax.set_title("Per-Document Retrieval Performance Breakdown")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=8.5, ncol=2)
    ax.grid(axis="y")
    save_fig(fig, os.path.join(out_dir, "per_document.png"))










def parse_args() -> Any:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    #input files
    ap.add_argument("--csv",           required=True,
                    help="Main results CSV (from eval_comprehensive.py).")
    ap.add_argument("--bootstrap_csv", default="",
                    help="Bootstrap CI CSV. Adds error bands.")
    ap.add_argument("--epoch_csv",     default="", help="Epoch comparison CSV.")
    ap.add_argument("--window_csvs",   nargs="*", default=[],
                    help="One CSV per window size, e.g. window_w5.csv.")
    #output
    ap.add_argument("--out_dir", default="plots_v3", help="Directory to save plots.")
    #model names
    ap.add_argument("--lmk_ft_model", default="LMK_FT")
    ap.add_argument("--base_lmk",    default="BASE_LMK")
    ap.add_argument("--base_cls",    default="BASE_CLS")
    ap.add_argument("--base_mean",   default="BASE_MEAN")
    ap.add_argument("--trained_cls",  default="CLS_FT16")
    ap.add_argument("--trained_mean", default="MEAN_FT16")
    # A
    ap.add_argument("--a1_model", default="A1_inbatch")
    ap.add_argument("--a2_model", default="A2_mined")
    ap.add_argument("--a3_model", default="A3_both")
    ap.add_argument("--a4_model", default="A4_random")
    # B
    ap.add_argument("--b1_model", default="B1_lmk_only")
    ap.add_argument("--b2_model", default="B2_last2")
    ap.add_argument("--b3_model", default="B3_last4")
    ap.add_argument("--b4_model", default="B4_full")
    #temp
    ap.add_argument("--d_temp001", default="D_temp001")
    ap.add_argument("--d_temp005", default="D_temp005")
    ap.add_argument("--d_temp01",  default="D_temp01")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: --csv not found: {args.csv}", file=sys.stderr)
        sys.exit(1)


    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nLoading {args.csv}")
    df           = load_csv(args.csv)
    models_found = df["model"].unique().tolist() if "model" in df.columns else []
    print(f"  Models:   {models_found}")
    print(f"  Columns:  {[c for c in df.columns if c not in ('model','paper_id','pooling','reranked')]}")


    bdf = load_bootstrap(args.bootstrap_csv)
    if not bdf.empty:
        print(f"  CI models: {bdf['model'].unique().tolist()}")
    else:
        print("  No bootstrap CSV -- CI bands disabled.")

    print(f"\nSaving plots to: {args.out_dir}/\n")

    print(" 1  Acc@K curves")
    plot_acc_at_k(df, bdf, args.out_dir)

    print(" 2  Negative strategy")
    plot_negative_strategy(df, bdf, args.out_dir, args)

    print(" 3  Layers unfrozen")
    plot_layers_unfrozen(df, bdf, args.out_dir, args)

    print(" 4  Acc vs epoch")
    plot_acc_vs_epoch(args.epoch_csv, df, bdf, args.out_dir, args)

    print(" 5  Window size")
    plot_window_size(args.window_csvs, args.out_dir)

    print(" 6  Pooling comparison")
    plot_pooling_comparison(df, bdf, args.out_dir, args)

    print(" 7  MRR comparison")
    plot_mrr(df, args.out_dir)

    print(" 8  Temperature sweep")
    plot_temperature(df, bdf, args.out_dir, args)

    print(" 9  Recall@K (LMK vs base MiniLM)")
    plot_recall_at_k(df, bdf, args.out_dir, args)

    print("10  Per-document breakdown")
    plot_per_document(df, bdf, args.out_dir, args)

    print(f"\nDone.  Plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
