#!/usr/bin/env python3
"""plot_embedding_shift.py — visualise per-query score shift from base to FT

— scatter of base_pos vs ft_pos per query (above diagonal = improved)
— KDE overlay of positive & hard-neg distributions before/after
— histogram of delta-margin per query


python3 plot_embedding_shift.py \\
    --base_model BASE_LMK:sentence-transformers/all-MiniLM-L6-v2:lmk \\
    --ft_model   B4_full:ft_runs/B4_full_finetune/epoch_5:lmk \\
    --paper_range 2-13 \\
    --out_plot plots_v3/embedding_shift.png
"""
from __future__ import annotations
import argparse
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib, platform
if platform.system() != "Darwin":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import gaussian_kde

# colours
BASE_COL = "#0072B2"   # blue
FT_COL = "#D55E00"  # orange

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 9,
    "lines.linewidth": 1.6,
})




# ---- encoder ----

class Encoder:
    """Minimal bi-encoder wrapper supporting lmk / cls / mean pooling"""

    def __init__(self, model_path: str, pooling: str, device: str,
                 max_length: int = 512):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.pooling_mode = pooling
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if pooling == "lmk":
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<LMK>"]}
            )
            self.tokenizer.truncation_side = "left"
            self.tokenizer.padding_side = "right"
        self._model = AutoModel.from_pretrained(model_path).to(device)
        if pooling == "lmk":
            self._model.resize_token_embeddings(len(self.tokenizer))
        self._model.eval()
        self._lmk_id = (
            self.tokenizer.convert_tokens_to_ids("<LMK>")
            if pooling == "lmk" else None
        )



    def embed(self, texts: List[str]) -> np.ndarray:
        import torch, torch.nn.functional as F
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            h = self._model(**enc).last_hidden_state  #[B, T, H]
        if self.pooling_mode == "cls":
            vecs = h[:, 0, :]
        elif self.pooling_mode == "mean":
            mask = enc["attention_mask"].unsqueeze(-1).float()
            vecs = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:  #lmk
            ids  = enc["input_ids"]
            lmk  = (ids == self._lmk_id).float()
            pos  = lmk * torch.arange(ids.shape[1], device=self.device).float()
            last = pos.argmax(dim=1)
            vecs = h[torch.arange(len(texts), device=self.device), last]
        return F.normalize(vecs, dim=-1).cpu().numpy()

    def format_query(self, q: str) -> str:
        return f"Question: {q} <LMK>" if self.pooling_mode == "lmk" \
               else f"Question: {q}"


def _cosine(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity between vector a [H] and matrix B [N, H]."""
    return (B @ a).astype(np.float32)   #already L2-normalised




# ---- per-query score gathering ----

def gather_per_query(
    enc: Encoder,
    paper_ids: List[int],
    *,
    doc_prefix: str = "TEST_PAPER",
    doc_ext: str  = ".mmd",
    qa_prefix: str = "QA_RAW",
    qa_var: str  = "QA_RAW",
    window_size: int = 10,
    n_hard_neg: int = 10,
    seed: int  = 42,
) -> List[Dict[str, float]]:
    """Returns one dict per query"""
    import lmk_train_data as D
    from lmk_train_data import build_doc_windows

    rng = random.Random(seed)
    results = []

    for pid in paper_ids:
        doc_path = f"{doc_prefix}{pid}{doc_ext}"
        qa_mod = f"{qa_prefix}{pid}"
        try:
            text = D.load_text_file(doc_path)
        except FileNotFoundError:
            print(f"[skip] {doc_path} not found"); continue
        try:
            qa_raw = D.load_qa_raw(qa_mod, qa_var)
        except (ImportError, AttributeError) as e:
            print(f"[skip] QA load error paper {pid}: {e}"); continue

        use_lmk = (enc.pooling_mode == "lmk")
        doc_idx = build_doc_windows(
            doc_id=f"{doc_prefix}{pid}", raw_text=text,
            tokenizer=enc.tokenizer, max_length=enc.max_length,
            window_size=window_size, landmark_token="<LMK>",
            use_lmk=use_lmk,
        )
        if not doc_idx.sentences:
            continue

        anchor_vecs = enc.embed(doc_idx.window_texts)   #[N, H]
        n_sent      = len(doc_idx.sentences)

        for item in qa_raw:
            q = str(item.get("question", "")).strip()
            golds = D.gold_to_ints(item.get("gold_sentence_idx", None))
            if not q or not golds:
                continue

            q_vec = enc.embed([enc.format_query(q)])[0]
            scores  = _cosine(q_vec, anchor_vecs)

            gold_set = set(golds)
            pos_vals = [float(scores[g]) for g in golds if 0 <= g < n_sent]
            non_gold = sorted(
                [float(scores[i]) for i in range(n_sent) if i not in gold_set],
                reverse=True,
            )
            if not pos_vals or not non_gold:
                continue

            results.append({
                "paper_id": pid,
                "pos": float(np.mean(pos_vals)),
                "hard_neg": float(np.mean(non_gold[:n_hard_neg])),
            })


    print(f"-> {len(results)} queries collected")
    return results


# -- plotting ---

def _kde(vals: List[float], xs: np.ndarray) -> np.ndarray:
    if len(vals) < 5:
        return np.zeros_like(xs)
    return gaussian_kde(vals, bw_method=0.15)(xs)


def plot_shift(
    base_data: List[Dict],
    ft_data: List[Dict],
    base_name: str,
    ft_name: str,
    out_path:str,
) -> None:
    #align queries -- take the shorter of the two lists
    n = min(len(base_data), len(ft_data))
    base_pos  = np.array([d["pos"]      for d in base_data[:n]])
    base_neg  = np.array([d["hard_neg"] for d in base_data[:n]])
    ft_pos    = np.array([d["pos"]      for d in ft_data[:n]])
    ft_neg    = np.array([d["hard_neg"] for d in ft_data[:n]])

    base_margin  = base_pos - base_neg
    ft_margin    = ft_pos   - ft_neg
    delta_margin = ft_margin - base_margin

    xs = np.linspace(0.0, 1.0, 400)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))


    # --scatter base_pos vs ft_pos--
    ax = axes[0]
    improve = ft_pos > base_pos
    ax.scatter(base_pos[~improve], ft_pos[~improve],
               color=BASE_COL, alpha=0.35, s=12)
    ax.scatter(base_pos[improve],  ft_pos[improve],
               color=FT_COL,   alpha=0.35, s=12)
    lims = [min(base_pos.min(), ft_pos.min()) - 0.02,
            max(base_pos.max(), ft_pos.max()) + 0.02]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)

    # --KDE overlay--
    ax = axes[1]
    ax.fill_between(xs, _kde(base_pos.tolist(), xs), alpha=0.20, color=BASE_COL)
    ax.plot(xs, _kde(base_pos.tolist(), xs), color=BASE_COL, linewidth=1.6)
    ax.fill_between(xs, _kde(ft_pos.tolist(), xs), alpha=0.20, color=FT_COL)
    ax.plot(xs, _kde(ft_pos.tolist(), xs), color=FT_COL, linewidth=1.6)
    # hard-ne
    ax.plot(xs, _kde(base_neg.tolist(), xs), color=BASE_COL, linewidth=1.2,
            linestyle="--")
    ax.plot(xs, _kde(ft_neg.tolist(), xs), color=FT_COL, linewidth=1.2,
            linestyle="--")
    # adding mean shift arrows
    ax.annotate("", xy=(ft_pos.mean(), 3.5), xytext=(base_pos.mean(), 3.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    ax.annotate("", xy=(ft_neg.mean(), 2.8), xytext=(base_neg.mean(), 2.8),
                arrowprops=dict(arrowstyle="->", color="grey", lw=1.0))

    # ----delta-margin histogram----
    ax = axes[2]
    counts, edges, patches = ax.hist(
        delta_margin, bins=40, color=FT_COL, alpha=0.75, edgecolor="white"
    )
    for patch, left in zip(patches, edges[:-1]):
        patch.set_facecolor(FT_COL if left >= 0 else BASE_COL)
        patch.set_alpha(0.7)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(delta_margin.mean(), color=FT_COL, linewidth=1.4, linestyle="-")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  ✓  {out_path}")




#------------
# --- main ---
#------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model",   required=True,
                    help="name:path:pooling  e.g. BASE_LMK:sentence-transformers/all-MiniLM-L6-v2:lmk")
    ap.add_argument("--ft_model",     required=True,
                    help="name:path:pooling  e.g. B4_full:ft_runs/B4_full_finetune/epoch_5:lmk")
    ap.add_argument("--paper_range",  default="2-13")
    ap.add_argument("--window_size",  type=int, default=10)
    ap.add_argument("--max_length",   type=int, default=512)
    ap.add_argument("--n_hard_neg",   type=int, default=10)
    ap.add_argument("--device",       default="")
    ap.add_argument("--out_plot",     default="plots_v3/embedding_shift.png")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    def parse(s):
        parts = s.split(":")
        name, pooling = parts[0], parts[-1]
        path = ":".join(parts[1:-1])
        return name, path, pooling

    lo, hi   = args.paper_range.split("-")
    paper_ids = list(range(int(lo), int(hi) + 1))

    base_name, base_path, base_pool = parse(args.base_model)
    ft_name,   ft_path,   ft_pool   = parse(args.ft_model)

    print(f"\n>>> Loading base encoder: {base_name}")
    base_enc = Encoder(base_path, base_pool, device, args.max_length)

    print(f">>> Gathering base scores ...")
    base_data = gather_per_query(
        base_enc, paper_ids,
        window_size=args.window_size,
        n_hard_neg=args.n_hard_neg,
    )

    del base_enc
    import torch; torch.cuda.empty_cache() if device != "cpu" else None

    print(f"\n>>> Loading fine-tuned encoder: {ft_name}")
    ft_enc = Encoder(ft_path, ft_pool, device, args.max_length)

    print(f">>> Gathering fine-tuned scores ...")
    ft_data = gather_per_query(
        ft_enc, paper_ids,
        window_size=args.window_size,
        n_hard_neg=args.n_hard_neg,
    )

    print(f"\n>>> Plotting ...")
    plot_shift(base_data, ft_data, base_name, ft_name, args.out_plot)
    print(">>> Done.")


if __name__ == "__main__":
    main()
