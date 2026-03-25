
"""plot_tsne.py — T-SNE  comparing base vs fine-tuned


python3 plot_tsne.py \\
    --base_model BASE_LMK:sentence-transformers/all-MiniLM-L6-v2:lmk \\
    --ft_model   B4_full:ft_runs/B4_full_finetune/epoch_5:lmk \\
    --paper_range 2-13 --color_by section --out_dir plots_v3/
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib
import platform
if platform.system() != "Darwin":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# --- colours ---
OKI = dict(
    blue = "#0072B2",
    vermillion = "#D55E00",
    green  = "#009E73",
    amber = "#E69F00",
    sky  = "#56B4E9",
    pink = "#CC79A7",
    yellow = "#F0E442",
    black = "#000000",
)

SECTION_COLORS = {
    "Early": OKI["green"],
    "Middle": OKI["amber"],
    "Late": OKI["vermillion"],
}

PAPER_PALETTE = [
    OKI["blue"], OKI["vermillion"], OKI["green"], OKI["amber"],
    OKI["sky"], OKI["pink"], "#332288", "#117733", "#882255", "#44AA99", "#999933","#DDCC77",
]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi":300,
    "font.family":"DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 8,
    "legend.framealpha": 0.85,
})


# --- parse model---

def parse_model_spec(s: str) -> Tuple[str, str, str]:
    parts = s.split(":")
    if len(parts) < 3:
        raise ValueError(
            f"Model spec must be 'name:path:pooling', got: {s!r}\n"
            "Example: BASE_LMK:sentence-transformers/all-MiniLM-L6-v2:lmk"
        )

    pooling  = parts[-1].strip()
    name  = parts[0].strip()
    model_path = ":".join(parts[1:-1]).strip()
    return name, model_path, pooling


# --- -embeddings ---

def build_embeddings(
    enc: Any,
    paper_ids: List[int],
    *,
    doc_prefix: str,
    doc_ext: str,
    qa_prefix: str,
    qa_var: str,
    window_size: int,
    max_length: int,
    include_queries: bool,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Embed all sentence windows """
    import lmk_train_data as D
    from lmk_train_data import build_doc_windows

    all_vecs: List[np.ndarray] = []
    all_meta: List[Dict[str, Any]] = []


    for pid in paper_ids:
        doc_path  = f"{doc_prefix}{pid}{doc_ext}"
        qa_module = f"{qa_prefix}{pid}"

        try:
            text = D.load_text_file(doc_path)
        except FileNotFoundError:
            print(f"  [skip] {doc_path} not found")
            continue

        use_lmk = (enc.pooling_mode == "lmk")
        doc_idx = build_doc_windows(
            doc_id=f"{doc_prefix}{pid}",
            raw_text=text,
            tokenizer=enc.tokenizer,
            max_length=max_length,
            window_size=window_size,
            landmark_token="<LMK>",
            use_lmk=use_lmk,
        )
        if not doc_idx.sentences:
            print(f"  [skip] paper {pid}: no sentences")
            continue

        n = len(doc_idx.sentences)

        # sentence embeddings
        vecs = enc.embed(doc_idx.window_texts)  #[n, H]
        for i, v in enumerate(vecs):
            frac    = i / max(1, n - 1)
            section = (
                "Early"  if frac < 1.0 / 3.0 else
                "Late"   if frac > 2.0 / 3.0 else
                "Middle"
            )
            snippet = doc_idx.sentences[i][:60] + ("…" if len(doc_idx.sentences[i]) > 60 else "")
            all_vecs.append(v)
            all_meta.append({
                "type":     "sentence",
                "paper_id": pid,
                "sent_idx": i,
                "n_sent":   n,
                "section":  section,
                "frac":     frac,
                "snippet":  snippet,
            })

        print(f"  paper {pid}: {n} sentence embeddings", end="", flush=True)

        #optional query embeddings
        if include_queries:
            try:
                qa_raw = D.load_qa_raw(qa_module, qa_var)
            except (ImportError, AttributeError):
                qa_raw = []
            q_texts: List[str] = []
            for item in qa_raw:
                q = str(item.get("question", "")).strip()
                if q:
                    q_texts.append(enc.format_query(q))
            if q_texts:
                q_vecs = enc.embed(q_texts)  #[Q, H]
                for j, qv in enumerate(q_vecs):
                    all_vecs.append(qv)
                    all_meta.append({
                        "type":     "query",
                        "paper_id": pid,
                        "sent_idx": -1,
                        "n_sent":   n,
                        "section":  "Query",
                        "frac":     -1.0,
                        "snippet":  q_texts[j][:60],
                    })
                print(f" + {len(q_vecs)} queries", end="", flush=True)
        print()

    if not all_vecs:
        return np.zeros((0, 0), dtype=np.float32), []

    return np.stack(all_vecs, axis=0).astype(np.float32), all_meta



# --- tsne ---


def fit_tsne_paired(
    base_embs: np.ndarray,
    ft_embs: np.ndarray,
    *,
    perplexity: int = 30,
    n_iter: int    = 1000,
    seed: int      = 42,
    metric: str    = "cosine",
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit T-SNE on the combinded [base; ft] matrix so panels share the same space.

    Returns (base_xy, ft_xy) each of shape [N, 2].
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print(
            "ERROR: scikit-learn is required for T-SNE.\n"
            "  Install with:  pip install scikit-learn",
            file=sys.stderr,
        )
        sys.exit(1)

    n_base   = len(base_embs)
    combined = np.vstack([base_embs, ft_embs]).astype(np.float32)

    #L2 normalise before cosine
    norms    = np.linalg.norm(combined, axis=1, keepdims=True)
    combined = combined / (norms + 1e-12)

    # clamp perplexity to val
    effective_perp = min(perplexity, (len(combined) - 1) // 3)
    effective_perp = max(effective_perp, 5)

    print(
        f"  Running T-SNE on {len(combined)} points "
        f"(perplexity={effective_perp}, n_iter={n_iter}) ..."
    )
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perp,
        max_iter=n_iter,
        metric=metric,
        random_state=seed,
        init="random",
        learning_rate="auto",
    )
    xy = tsne.fit_transform(combined)  #[2N, 2]

    #
    # print(f"tsne done: base_xy={xy[:n_base].shape}  ft_xy={xy[n_base:].shape}")

    return xy[:n_base], xy[n_base:]



# --- helper funcs ---

def _scatter_section(
    ax: Any,
    xy: np.ndarray,
    meta: List[Dict[str, Any]],
    title: str,
) -> None:
    """scatter by document section"""
    for section, color in SECTION_COLORS.items():
        mask = [m["section"] == section and m["type"] == "sentence" for m in meta]
        idx = [i for i, m in enumerate(mask) if m]
        if idx:
            ax.scatter(
                xy[idx, 0], xy[idx, 1],
                c=color, s=8, alpha=0.55, label=section, linewidths=0,
            )
    #queries as distinct markers
    q_idx = [i for i, m in enumerate(meta) if m.get("type") == "query"]
    if q_idx:
        ax.scatter(
            xy[q_idx, 0], xy[q_idx, 1],
            marker="x", c=OKI["black"], s=25, alpha=0.6,
            linewidths=0.8, label="Query", zorder=5,
        )
    ax.set_title(title)
    ax.set_xlabel("T-SNE dim 1")
    ax.set_ylabel("T-SNE dim 2")
    ax.legend(markerscale=2, fontsize=7.5, loc="best")
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    ax.set_aspect("equal", adjustable="datalim")




def _scatter_paper(
    ax: Any,
    xy: np.ndarray,
    meta: List[Dict[str, Any]],
    paper_ids: List[int],
    title: str,
) -> None:
    """Scatter coloured by paper"""
    pid_to_color = {pid: PAPER_PALETTE[i % len(PAPER_PALETTE)]
                    for i, pid in enumerate(sorted(set(paper_ids)))}
    for pid, color in pid_to_color.items():
        idx = [i for i, m in enumerate(meta)
               if m["paper_id"] == pid and m["type"] == "sentence"]
        if idx:
            ax.scatter(
                xy[idx, 0], xy[idx, 1],
                c=color, s=8, alpha=0.50, linewidths=0,
                label=f"P{pid}",
            )

    q_idx = [i for i, m in enumerate(meta) if m.get("type") == "query"]
    if q_idx:
        ax.scatter(
            xy[q_idx, 0], xy[q_idx, 1],
            marker="x", c=OKI["black"], s=25, alpha=0.6,
            linewidths=0.8, label="Query", zorder=5,
        )
    ax.set_title(title)
    ax.set_xlabel("T-SNE dim 1")
    ax.set_ylabel("T-SNE dim 2")
    ax.legend(markerscale=2, fontsize=6.5, loc="best",
              ncol=max(1, len(pid_to_color) // 6))
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    ax.set_aspect("equal", adjustable="datalim")


def plot_tsne(
    base_xy: np.ndarray,
    ft_xy: np.ndarray,
    base_meta: List[Dict[str, Any]],
    ft_meta: List[Dict[str, Any]],
    base_name: str,
    ft_name: str,
    paper_ids: List[int],
    color_by: str,
    out_path: str,
) -> None:

    if color_by == "both":
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        _scatter_section(axes[0][0], base_xy, base_meta, f"Section — {base_name}")
        _scatter_section(axes[0][1], ft_xy, ft_meta, f"Section — {ft_name}")
        _scatter_paper(  axes[1][0], base_xy, base_meta, paper_ids, f"Paper — {base_name}")
        _scatter_paper(  axes[1][1], ft_xy,   ft_meta,   paper_ids, f"Paper — {ft_name}")

    elif color_by == "paper":
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        _scatter_paper(axes[0], base_xy, base_meta, paper_ids, f"Paper — {base_name}")
        _scatter_paper(axes[1], ft_xy,   ft_meta,   paper_ids, f"Paper — {ft_name}")

    else:  # section
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        _scatter_section(axes[0], base_xy, base_meta, f"Section — {base_name}")
        _scatter_section(axes[1], ft_xy,   ft_meta,   f"Section — {ft_name}")

    fig.suptitle(
        "T-SNE of Sentence Embeddings: Before vs After Fine-Tuning\n"
        "(Both panels fitted on concatenated embeddings — positions are comparable)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(pad=0.5, rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    #print(f"{out_path}")
    print(f"{out_path}")

#---------------------
# ------- main ------
#---------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--base_model", required=True,
        help="Baseline model spec: 'name:path:pooling'.  "
             "eg. BASE_LMK:sentence-transformers/all-MiniLM-L6-v2:lmk",
    )
    ap.add_argument(
        "--ft_model", required=True,
        help="fine-tuned model spec 'name:path:pooling'.",
    )
    ap.add_argument("--paper_range", default="2-13",
                    help="Paper IDs, eg.'2-13' or '2,5,7-10'.")
    ap.add_argument("--doc_prefix", default="TEST_PAPER")
    ap.add_argument("--doc_ext", default=".mmd")
    ap.add_argument("--qa_prefix", default="QA_RAW")
    ap.add_argument("--qa_var",default="QA_RAW")
    ap.add_argument("--window_size", type=int, default=10)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument(
        "--color_by", choices=["section", "paper", "both"], default="section",
        help="Colour encoding for scatter points.",
    )
    ap.add_argument(
        "--include_queries", action="store_true",
        help="Overlay query embeddings as 'x' markers.",
    )

    ap.add_argument("--perplexity",     type=int, default=30,
                    help="T-SNE perplexity (auto-clamped if too large).")
    ap.add_argument("--n_iter",         type=int, default=1000,
                    help="T-SNE iterations.")
    ap.add_argument("--seed",           type=int, default=42)
    ap.add_argument("--device",         default="",
                    help="Torch device (auto-detected if empty).")
    ap.add_argument("--out_dir",        default="plots_v3",
                    help="Output directory.")

    args = ap.parse_args()

    try:
        from LMK_Embed10_DENSE_ONLY import Encoder
        import lmk_train_data as D
    except ImportError as e:
        print(f"ERROR: couldnt import pipeline modules: {e}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    paper_ids = D.parse_int_range(args.paper_range)
    if not paper_ids:
        print("ERROR: --paper_range produced no paper IDs.", file=sys.stderr)
        sys.exit(1)
    print(f"\n>>> Papers: {paper_ids}")

    base_name, base_path, base_pooling = parse_model_spec(args.base_model)
    ft_name,   ft_path,   ft_pooling   = parse_model_spec(args.ft_model)
    device = args.device if args.device else None

    # baseline encoder
    print(f"\n>>> Loading baseline encoder: {base_name}")
    base_enc = Encoder(
        model_name=base_path, pooling_mode=base_pooling,
        batch_size=args.batch_size, max_length=args.max_length, device=device,
    )
    print(">>> Building baseline embeddings")
    base_embs, base_meta = build_embeddings(
        base_enc, paper_ids,
        doc_prefix=args.doc_prefix, doc_ext=args.doc_ext,
        qa_prefix=args.qa_prefix, qa_var=args.qa_var,
        window_size=args.window_size, max_length=args.max_length,
        include_queries=args.include_queries,
    )
    if base_embs.shape[0] == 0:
        print("ERROR: No baseline embeddings built", file=sys.stderr)
        sys.exit(1)
    print(f"Baseline: {base_embs.shape[0]} points, dim={base_embs.shape[1]}")

    # fine-tuned encoder
    print(f"\n>>> Loading fine-tuned encoder: {ft_name}")
    ft_enc = Encoder(
        model_name=ft_path, pooling_mode=ft_pooling,
        batch_size=args.batch_size, max_length=args.max_length, device=device,
    )
    print(">>> Building fine-tuned embeddings ...")
    ft_embs, ft_meta = build_embeddings(
        ft_enc, paper_ids,
        doc_prefix=args.doc_prefix, doc_ext=args.doc_ext,
        qa_prefix=args.qa_prefix, qa_var=args.qa_var,
        window_size=args.window_size, max_length=args.max_length,
        include_queries=args.include_queries,
    )
    if ft_embs.shape[0] == 0:
        print("ERROR: No fine-tuned embeddings built", file=sys.stderr)
        sys.exit(1)
    print(f"Fine-tuned: {ft_embs.shape[0]} points, dim={ft_embs.shape[1]}")

    # warn if sizes differ
    if base_embs.shape[0] != ft_embs.shape[0]:
        print(
            f"Embedding counts differ "
            f"({base_embs.shape[0]} vs {ft_embs.shape[0]}).  "
            "Ensure both models evaluate the same papers."
        )

    print("\n>>> Fitting T-SNE ...")
    base_xy, ft_xy = fit_tsne_paired(
        base_embs, ft_embs,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        seed=args.seed,
    )
    print(f" T-SNE complete.  base_xy={base_xy.shape}  ft_xy={ft_xy.shape}")

    print("\n>>> Plotting ...")
    out_path = os.path.join(args.out_dir, "tsne_embeddings.png")
    plot_tsne(
        base_xy, ft_xy, base_meta, ft_meta,
        base_name, ft_name, paper_ids,
        color_by=args.color_by,
        out_path=out_path,
    )

    print("\n>>> Done.")
    print(f"    T-SNE plot saved to: {out_path}")
    



if __name__ == "__main__":
    main()
