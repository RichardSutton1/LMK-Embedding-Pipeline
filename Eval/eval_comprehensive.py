"""eval_comprehensive.py — multi-model, multi-paper LMK retrieval evaluation

Evaluates dense retrieval (optional CE)
Acc@K (anchor + span), MRR (anchor + span), and NDCG@K (span) for both dense-only and CE-reranked rankings



python eval_comprehensive.py \\
    --paper_range 1-24 \\
    --model LMK_FT:ft_runs/epoch_3:lmk \\
    --model CLS_BASE:sentence-transformers/all-MiniLM-L6-v2:cls \\
    --ce_model cross-encoder/ms-marco-MiniLM-L-6-v2 \\
    --k_max 10 --out_csv results.csv

"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

import lmk_train_data as D

from LMK_Embed10_DENSE_ONLY import (
    Encoder,
    CrossEncoderReranker,
    build_sentence_index_dense_only,
    cosine_scores_matrix,
)


# ---- model spec parsing ----

@dataclass
class ModelSpec:
    name: str
    model_path: str
    pooling: str  # "lmk" | "cls" | "mean"


def parse_model_spec(s: str) -> ModelSpec:
    """Parse 'name:path:pooling' into a ModelSpec. Path may contain slashes."""
    parts = s.split(":")
    if len(parts) < 3:
        raise ValueError(
            f"Model spec must be 'name:path:pooling', got: {s!r}\n"
            "Example: LMK_FT:ft_runs/epoch_3:lmk"
        )
    pooling = parts[-1].strip()
    name = parts[0].strip()
    model_path = ":".join(parts[1:-1]).strip()
    if pooling not in Encoder.POOLING_MODES:
        raise ValueError(f"pooling must be one of {Encoder.POOLING_MODES}, got {pooling!r}")
    return ModelSpec(name=name, model_path=model_path, pooling=pooling)


# --- per-question result container ---

@dataclass
class QuestionResult:
    qid: str
    question: str
    gold_idxs: List[int]
    n_sent: int
    # dense-only ranking
    dense_rank_anchor: int  # 1-based; k_max+1 if not found in top k_max
    dense_rank_span: int  # 1-based (span-deduped)
    dense_acc_anchor: List[int] #[acc@1, acc@2, ..., acc@k_max]
    dense_acc_span: List[int]
    #CE-reranked (none if no reranker)
    ce_rank_anchor: Optional[int] = None
    ce_rank_span: Optional[int] = None
    ce_acc_anchor: Optional[List[int]] = None
    ce_acc_span: Optional[List[int]] = None




# ---helpers ----

def _dedup_spans(
    candidates: List[Dict[str, Any]],
    sentences: List[str],
    span_pre: int,
    span_post: int,
) -> List[Dict[str, Any]]:
    """returns candidates in score order with duplicate spans removed"""
    n_sent = len(sentences)
    seen: set = set()
    deduped: List[Dict[str, Any]] = []

    for c in candidates:
        i = int(c["sentence_idx"])
        lo = max(0, i - span_pre)
        hi = min(n_sent - 1, i + span_post)
        key = (lo, hi)
        if key not in seen:
            seen.add(key)
            deduped.append({**c, "span_lo": lo, "span_hi": hi})

    return deduped



def _ranks(
    candidates: List[Dict[str, Any]],
    gold_idxs: List[int],
    sentences: List[str],
    span_pre: int,
    span_post: int,
    k_max: int,
) -> Tuple[int, int, List[int], List[int]]:
    """Computes anchor and span ranks + Acc@K arrays for one question
    Args:
    Returns: (rank_anchor, rank_span, acc_anchor, acc_span)
    """
    gold_set = set(gold_idxs)
    k_candidates = candidates[:k_max]

    #first position where anchor idx in gold_set
    rank_anchor = k_max + 1
    for i, c in enumerate(k_candidates):
        if int(c["sentence_idx"]) in gold_set:
            rank_anchor = i + 1
            break

    #deduplicate, find first span covering any gold
    deduped = _dedup_spans(candidates, sentences, span_pre, span_post)
    k_deduped = deduped[:k_max]

    rank_span = k_max + 1
    for i, c in enumerate(k_deduped):
        if any(int(c["span_lo"]) <= g <= int(c["span_hi"]) for g in gold_idxs):
            rank_span = i + 1
            break

    #acc@k arrays
    acc_anchor = [0] * k_max
    acc_span = [0] * k_max
    found_anchor = False
    found_span = False

    for k in range(k_max):
        if k < len(k_candidates):
            found_anchor = found_anchor or (int(k_candidates[k]["sentence_idx"]) in gold_set)
        if k < len(k_deduped):
            lo, hi = int(k_deduped[k]["span_lo"]), int(k_deduped[k]["span_hi"])
            found_span = found_span or any(lo <= g <= hi for g in gold_idxs)
        acc_anchor[k] = 1 if found_anchor else 0
        acc_span[k] = 1 if found_span else 0

    return rank_anchor, rank_span, acc_anchor, acc_span




def _mrr(rank: int, k_max: int) -> float:
    return 1.0 / rank if rank <= k_max else 0.0






def _ndcg(rank: int, k_max: int) -> float:
    """Binary-relevance NDCG@k_max with IDCG=1 (single positive)."""
    if rank > k_max:
        return 0.0
    return 1.0 / math.log2(rank + 1)





# --- aggregate metrics ---

@dataclass
class PaperMetrics:
    n_questions: int
    #dense
    acc_anchor: List[float]
    acc_span: List[float]
    mrr_anchor: float
    mrr_span: float
    ndcg_span: float
    no_hits: int
    # CE
    ce_acc_anchor: Optional[List[float]] = None
    ce_acc_span: Optional[List[float]] = None
    ce_mrr_anchor: Optional[float] = None
    ce_mrr_span: Optional[float] = None
    ce_ndcg_span: Optional[float] = None


def aggregate_results(results: List[QuestionResult], k_max: int) -> PaperMetrics:
    n = len(results)
    if n == 0:
        empty = [0.0] * k_max
        return PaperMetrics(0, empty, empty, 0.0, 0.0, 0.0, 0)

    has_ce = results[0].ce_rank_anchor is not None

    acc_anchor = [sum(r.dense_acc_anchor[k] for r in results) / n for k in range(k_max)]
    acc_span = [sum(r.dense_acc_span[k] for r in results) / n for k in range(k_max)]
    mrr_anchor = sum(_mrr(r.dense_rank_anchor, k_max) for r in results) / n
    mrr_span = sum(_mrr(r.dense_rank_span, k_max) for r in results) / n
    ndcg_span = sum(_ndcg(r.dense_rank_span, k_max) for r in results) / n
    no_hits = sum(1 for r in results if r.dense_rank_anchor == k_max + 1
                  and r.dense_rank_span == k_max + 1)

    m = PaperMetrics(
        n_questions=n,
        acc_anchor=acc_anchor, acc_span=acc_span,
        mrr_anchor=mrr_anchor, mrr_span=mrr_span,
        ndcg_span=ndcg_span, no_hits=no_hits,
    )

    if has_ce:
        m.ce_acc_anchor = [
            sum(r.ce_acc_anchor[k] for r in results if r.ce_acc_anchor is not None) / n  # FIX: is not None
            for k in range(k_max)
        ]
        m.ce_acc_span = [
            sum(r.ce_acc_span[k] for r in results if r.ce_acc_span is not None) / n  # FIX: is not None
            for k in range(k_max)
        ]
        m.ce_mrr_anchor = sum(
            _mrr(r.ce_rank_anchor, k_max) for r in results if r.ce_rank_anchor is not None
        ) / n
        m.ce_mrr_span = sum(
            _mrr(r.ce_rank_span, k_max) for r in results if r.ce_rank_span is not None
        ) / n
        m.ce_ndcg_span = sum(
            _ndcg(r.ce_rank_span, k_max) for r in results if r.ce_rank_span is not None
        ) / n

    return m


def micro_aggregate(per_paper: List[PaperMetrics], k_max: int) -> PaperMetrics:
    """Micro-average across papers (weighted by paper question count)."""
    total = sum(p.n_questions for p in per_paper)
    if total == 0:
        empty = [0.0] * k_max
        return PaperMetrics(0, empty, empty, 0.0, 0.0, 0.0, 0)

    has_ce = all(p.ce_acc_anchor is not None for p in per_paper) if per_paper else False

    def wavg(getter):
        return sum(getter(p) * p.n_questions for p in per_paper) / total

    acc_anchor = [wavg(lambda p, k=k: p.acc_anchor[k]) for k in range(k_max)]
    acc_span = [wavg(lambda p, k=k: p.acc_span[k]) for k in range(k_max)]

    m = PaperMetrics(
        n_questions=total,
        acc_anchor=acc_anchor,
        acc_span=acc_span,
        mrr_anchor=wavg(lambda p: p.mrr_anchor),
        mrr_span=wavg(lambda p: p.mrr_span),
        ndcg_span=wavg(lambda p: p.ndcg_span),
        no_hits=sum(p.no_hits for p in per_paper),
    )
    if has_ce:
        m.ce_acc_anchor = [wavg(lambda p, k=k: p.ce_acc_anchor[k]) for k in range(k_max)]  # type: ignore[index]
        m.ce_acc_span = [wavg(lambda p, k=k: p.ce_acc_span[k]) for k in range(k_max)]  # type: ignore[index]
        m.ce_mrr_anchor = wavg(lambda p: p.ce_mrr_anchor or 0.0)
        m.ce_mrr_span = wavg(lambda p: p.ce_mrr_span or 0.0)
        m.ce_ndcg_span = wavg(lambda p: p.ce_ndcg_span or 0.0)
    return m


# --- printing helpers ---

def _fmt(x: float) -> str:
    return f"{x:.3f}"




def _print_model_table(
    model_name: str,
    paper_ids: List[int],
    per_paper: List[PaperMetrics],
    agg: PaperMetrics,
    k_max: int,
    show_ks: Tuple[int, ...] = (1, 3, 5, 10),
) -> None:
    ks = [k for k in show_ks if k <= k_max]
    header_ks = "  ".join(f"Acc_s@{k}" for k in ks)
    has_ce = per_paper[0].ce_acc_anchor is not None if per_paper else False

    print(f"\n{'─'*80}")
    print(f"MODEL: {model_name}")
    print(f"{'─'*80}")

    print(f"[DENSE-ONLY]")
    print(f"{'Paper':<10} {'N_q':>5}  {header_ks}  MRR_s   NDCG_s  Acc_a@1  MRR_a")
    for pid, pm in zip(paper_ids, per_paper):
        row_ks = "  ".join(_fmt(pm.acc_span[k - 1]) for k in ks)
        print(
            f"Paper {pid:<4} {pm.n_questions:>5}  {row_ks}  "
            f"{_fmt(pm.mrr_span)}  {_fmt(pm.ndcg_span)}  "
            f"{_fmt(pm.acc_anchor[0])}  {_fmt(pm.mrr_anchor)}"
        )
    row_ks = "  ".join(_fmt(agg.acc_span[k - 1]) for k in ks)
    print(
        f" {'AGGREGATE':<10} {agg.n_questions:>5}  {row_ks}  "
        f"{_fmt(agg.mrr_span)}  {_fmt(agg.ndcg_span)}  "
        f"{_fmt(agg.acc_anchor[0])}  {_fmt(agg.mrr_anchor)}"
    )

    if has_ce:
        print(f"\n[CE RERANKED]")
        print(f" {'Paper':<10} {'N_q':>5}  {header_ks}  MRR_s   NDCG_s  Acc_a@1  MRR_a")
        for pid, pm in zip(paper_ids, per_paper):
            row_ks = "  ".join(_fmt(pm.ce_acc_span[k - 1]) for k in ks)  
            print(
                f"  Paper {pid:<4} {pm.n_questions:>5}  {row_ks}  "
                f"{_fmt(pm.ce_mrr_span or 0)}  {_fmt(pm.ce_ndcg_span or 0)}  "
                f"{_fmt(pm.ce_acc_anchor[0])}  {_fmt(pm.ce_mrr_anchor or 0)}"  
            )
        row_ks = "  ".join(_fmt(agg.ce_acc_span[k - 1]) for k in ks)  #type: ignore[index]
        print(
            f"  {'AGGREGATE':<10} {agg.n_questions:>5}  {row_ks}  "
            f"{_fmt(agg.ce_mrr_span or 0)}  {_fmt(agg.ce_ndcg_span or 0)}  "
            f"{_fmt(agg.ce_acc_anchor[0])}  {_fmt(agg.ce_mrr_anchor or 0)}" 
        )


# --- CSV / JSON helpers ---

def _metrics_row(
    model_name: str,
    pooling: str,
    paper_id: Any,
    pm: PaperMetrics,
    k_max: int,
    ce: bool,
) -> Dict[str, Any]:
    acc_a = pm.ce_acc_anchor if ce else pm.acc_anchor
    acc_s = pm.ce_acc_span if ce else pm.acc_span
    mrr_a = pm.ce_mrr_anchor if ce else pm.mrr_anchor
    mrr_s = pm.ce_mrr_span if ce else pm.mrr_span
    ndcg_s = pm.ce_ndcg_span if ce else pm.ndcg_span
    row: Dict[str, Any] = {
        "model": model_name,
        "pooling": pooling,
        "reranked": ce,
        "paper_id": paper_id,
        "n_questions": pm.n_questions,
        "no_hits": pm.no_hits if not ce else None,
        f"mrr_anchor": mrr_a,
        f"mrr_span": mrr_s,
        f"ndcg_span@{k_max}": ndcg_s,
    }

    for k in range(1, k_max + 1):
        row[f"acc_anchor@{k}"] = acc_a[k - 1] if acc_a else None
        row[f"acc_span@{k}"] = acc_s[k - 1] if acc_s else None

    return row





def write_csv(
    path: str,
    all_rows: List[Dict[str, Any]],
    k_max: int,
) -> None:
    fieldnames = (
        ["model", "pooling", "reranked", "paper_id", "n_questions", "no_hits",
         "mrr_anchor", "mrr_span", f"ndcg_span@{k_max}"]
        + [f"acc_anchor@{k}" for k in range(1, k_max + 1)]
        + [f"acc_span@{k}" for k in range(1, k_max + 1)]
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n[CSV] {path}")





# --- evaluate one model over all papers ---

def evaluate_model(
    spec: ModelSpec,
    paper_ids: List[int],
    *,
    doc_prefix: str,
    doc_ext: str,
    qa_prefix: str,
    qa_var: str,
    reranker: Optional[CrossEncoderReranker],
    window_size: int,
    max_length: int,
    batch_size: int,
    dense_top_n: int,
    ce_batch_size: int,
    span_pre: int,
    span_post: int,
    k_max: int,
    device: str = "",
) -> Tuple[List[int], List[PaperMetrics], List[List[QuestionResult]]]:
    """loads encoder, builds indexes, rusn eval over all papers"""
    # validate local checkpoint paths early
    looks_local = (         #HF IDs have 1 slash
        spec.model_path.startswith(".")
        or spec.model_path.startswith(os.sep) 
        or spec.model_path.count("/") != 1
    )
    if looks_local and not os.path.isdir(spec.model_path):
        raise FileNotFoundError(
            f"Model checkpoint directory not found: {spec.model_path!r}\n"
            f"  Run training first, then re-run eval."
        )

    print(f"\n>>> Loading encoder: {spec.name}  (model={spec.model_path}, pooling={spec.pooling})")
    enc = Encoder(
        model_name=spec.model_path,
        pooling_mode=spec.pooling,
        batch_size=batch_size,
        max_length=max_length,
        device=device if device else None,
    )

    evaluated_ids: List[int] = []
    per_paper_metrics: List[PaperMetrics] = []
    per_paper_results: List[List[QuestionResult]] = []

    for pid in paper_ids:
        doc_path = f"{doc_prefix}{pid}{doc_ext}"
        qa_module = f"{qa_prefix}{pid}"

        print(f"Paper {pid}: {doc_path}", end="", flush=True)

        try:
            text = D.load_text_file(doc_path)
        except FileNotFoundError:
            print(f"[SKIP — file not found]")
            continue

        try:
            qa_raw = D.load_qa_raw(qa_module, qa_var)
        except (ImportError, AttributeError) as e:
            print(f"[SKIP — QA load error: {e}]")
            continue

        # build index
        #use_lmk depends on pooling mode
        from lmk_train_data import build_doc_windows
        doc_idx = build_doc_windows(
            doc_id=f"{doc_prefix}{pid}",
            raw_text=text,
            tokenizer=enc.tokenizer,
            max_length=max_length,
            window_size=window_size,
            landmark_token="<LMK>",
            use_lmk=(spec.pooling == "lmk"),
        )
        sentences = doc_idx.sentences
        n_sent = len(sentences)
        if n_sent == 0:
            print("  [SKIP — no sentences]")
            continue

        #embedding all anchor windows
        anchor_vecs = enc.embed(doc_idx.window_texts) #[N, H]

        # print(f"  anchor_vecs shape: {anchor_vecs.shape}")
        print(f"  ({n_sent} sentences indexed)", flush=True)

        question_results: List[QuestionResult] = []

        for item in qa_raw:
            qid = str(item.get("id", "")).strip()
            question = str(item.get("question", "")).strip()
            golds = D.gold_to_ints(item.get("gold_sentence_idx", None))
            if not qid or not question or not golds:
                continue

            #ense
            q_text = enc.format_query(question)
            q_vec = enc.embed([q_text])[0]
            scores = cosine_scores_matrix(q_vec, anchor_vecs)
            top_idx = np.argsort(-scores)[: min(dense_top_n, n_sent)]
            dense_candidates = [
                {"sentence_idx": int(i), "dense_score": float(scores[i])}
                for i in top_idx
            ]

            ra, rs, aa, as_ = _ranks(
                dense_candidates, golds, sentences, span_pre, span_post, k_max
            )

            qr = QuestionResult(
                qid=qid, question=question, gold_idxs=golds, n_sent=n_sent,
                dense_rank_anchor=ra, dense_rank_span=rs,
                dense_acc_anchor=aa, dense_acc_span=as_,
            )

            # CE
            if reranker is not None:
                ce_candidates_with_spans = []
                deduped_for_ce = _dedup_spans(dense_candidates, sentences, span_pre, span_post)
                for c in deduped_for_ce[:dense_top_n]:
                    span_text = " ".join(sentences[c["span_lo"] : c["span_hi"] + 1])
                    ce_candidates_with_spans.append({**c, "span_text": span_text})

                pairs = [(question, c["span_text"]) for c in ce_candidates_with_spans]
                ce_scores = reranker.score_pairs(pairs, batch_size=ce_batch_size)

                for c, s in zip(ce_candidates_with_spans, ce_scores):
                    c["ce_score"] = float(s)
                ce_candidates_with_spans.sort(key=lambda c: c["ce_score"], reverse=True)

            
                ce_gold_set = set(golds)
                ce_rank_anchor = k_max + 1
                ce_rank_span = k_max + 1
                ce_acc_anchor_list = [0] * k_max
                ce_acc_span_list = [0] * k_max
                found_a = False
                found_s = False
                for k, c in enumerate(ce_candidates_with_spans[:k_max]):
                    found_a = found_a or (int(c["sentence_idx"]) in ce_gold_set)
                    found_s = found_s or any(
                        int(c["span_lo"]) <= g <= int(c["span_hi"]) for g in golds
                    )
                    if found_a and ce_rank_anchor == k_max + 1:
                        ce_rank_anchor = k + 1
                    if found_s and ce_rank_span == k_max + 1:
                        ce_rank_span = k + 1
                    ce_acc_anchor_list[k] = 1 if found_a else 0
                    ce_acc_span_list[k] = 1 if found_s else 0

                qr.ce_rank_anchor = ce_rank_anchor
                qr.ce_rank_span = ce_rank_span
                qr.ce_acc_anchor = ce_acc_anchor_list
                qr.ce_acc_span = ce_acc_span_list

            question_results.append(qr)

        pm = aggregate_results(question_results, k_max)
        evaluated_ids.append(pid)
        per_paper_metrics.append(pm)
        per_paper_results.append(question_results)

    return evaluated_ids, per_paper_metrics, per_paper_results





# --------------MAIN--------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # paper
    ap.add_argument("--paper_range", type=str, default="1-24",
                    help="Paper index range, e.g. '1-24' or '1-6,9,12-14'.")
    ap.add_argument("--paper_exclude", type=str, default="",
                    help="Comma-separated indices to exclude.")
    ap.add_argument("--doc_prefix", type=str, default="TEST_PAPER")
    ap.add_argument("--doc_ext", type=str, default=".mmd")
    ap.add_argument("--qa_prefix", type=str, default="QA_RAW")
    ap.add_argument("--qa_var", type=str, default="QA_RAW")

    # model
    ap.add_argument(
        "--model", action="append", dest="models", default=[], metavar="SPEC",
        help="Model spec: 'name:model_path:pooling'.  Repeatable.",
    )

    # ce
    ap.add_argument("--ce_model", type=str, default="",
                    help="CE model name/path. Omit to disable CE reranking.")
    ap.add_argument("--no_ce", action="store_true", help="Disable CE reranking.")

    # retrieval params
    ap.add_argument("--window_size", type=int, default=10)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dense_top_n", type=int, default=100)
    ap.add_argument("--span_pre", type=int, default=2)
    ap.add_argument("--span_post", type=int, default=1)
    ap.add_argument("--ce_batch_size", type=int, default=16)
    ap.add_argument("--k_max", type=int, default=10)

    
    ap.add_argument("--device", type=str, default="",
                    help="Torch device, e.g. 'cuda' or 'cpu'. Auto-detected if not set.")

    #out dir
    ap.add_argument("--out_csv", type=str, default="", help="Path for CSV results.")
    ap.add_argument("--out_json", type=str, default="", help="Path for JSON results.")

    args = ap.parse_args()

    if not args.models:
        ap.error("Provide at least one --model spec.")

    specs = [parse_model_spec(s) for s in args.models]

    include = D.parse_int_range(args.paper_range)
    exclude = set(D.parse_int_range(args.paper_exclude))
    paper_ids = [i for i in include if i not in exclude]
    if not paper_ids:
        ap.error("--paper_range produced no papers after exclude filter.")

    print(f">>> Papers: {paper_ids}")
    print(f">>> Models: {[f'{s.name}({s.pooling})' for s in specs]}")
    print(f">>> Span: pre={args.span_pre} post={args.span_post}  k_max={args.k_max}")

    reranker: Optional[CrossEncoderReranker] = None
    if args.ce_model and not args.no_ce:
        print(f">>> CE reranker: {args.ce_model}")
        reranker = CrossEncoderReranker(model_name=args.ce_model)

    all_csv_rows: List[Dict[str, Any]] = []
    json_output: Dict[str, Any] = {
        "config": {
            "paper_range": args.paper_range,
            "span_pre": args.span_pre, "span_post": args.span_post,
            "k_max": args.k_max, "dense_top_n": args.dense_top_n,
            "window_size": args.window_size, "max_length": args.max_length,
            "ce_model": args.ce_model if reranker else None,
        },
        "models": [],
    }

    for spec in specs:
        try:
            evaluated_paper_ids, per_paper_metrics, per_paper_results = evaluate_model(
                spec=spec,
                paper_ids=paper_ids,
                doc_prefix=args.doc_prefix,
                doc_ext=args.doc_ext,
                qa_prefix=args.qa_prefix,
                qa_var=args.qa_var,
                reranker=reranker,
                window_size=args.window_size,
                max_length=args.max_length,
                batch_size=args.batch_size,
                dense_top_n=args.dense_top_n,
                ce_batch_size=args.ce_batch_size,
                span_pre=args.span_pre,
                span_post=args.span_post,
                k_max=args.k_max,
                device=args.device,
            )
        except FileNotFoundError as e:
            print(f"\n  [ERROR] {e}")
            continue
        except Exception as e:
            print(f"  [SKIP] Model {spec.name} failed to evaluate: {e}")
            continue

        if not per_paper_metrics:
            print(f"  [WARNING] No papers evaluated for model {spec.name}")
            continue

        agg = micro_aggregate(per_paper_metrics, args.k_max)

        _print_model_table(
            model_name=f"{spec.name} ({spec.pooling})",
            paper_ids=evaluated_paper_ids,
            per_paper=per_paper_metrics,
            agg=agg,
            k_max=args.k_max,
        )

        #accumulate CSV rows
        for pid, pm in zip(evaluated_paper_ids, per_paper_metrics):
            all_csv_rows.append(_metrics_row(spec.name, spec.pooling, pid, pm, args.k_max, ce=False))
            if pm.ce_acc_span is not None:
                all_csv_rows.append(_metrics_row(spec.name, spec.pooling, pid, pm, args.k_max, ce=True))
        all_csv_rows.append(_metrics_row(spec.name, spec.pooling, "AGGREGATE", agg, args.k_max, ce=False))
        if agg.ce_acc_span is not None:
            all_csv_rows.append(_metrics_row(spec.name, spec.pooling, "AGGREGATE", agg, args.k_max, ce=True))

        #accumulate JSON
        model_json: Dict[str, Any] = {
            "name": spec.name,
            "model_path": spec.model_path,
            "pooling": spec.pooling,
            "papers": [],
            "aggregate": {},
        }
        for pid, pm, qrs in zip(evaluated_paper_ids, per_paper_metrics, per_paper_results):
            paper_entry: Dict[str, Any] = {
                "paper_id": pid,
                "n_questions": pm.n_questions,
                "no_hits": pm.no_hits,
                "dense": {
                    "acc_anchor_at_k": pm.acc_anchor,
                    "acc_span_at_k": pm.acc_span,
                    "mrr_anchor": pm.mrr_anchor,
                    "mrr_span": pm.mrr_span,
                    f"ndcg_span@{args.k_max}": pm.ndcg_span,
                },
            }
            if pm.ce_acc_span is not None:
                paper_entry["ce"] = {
                    "acc_anchor_at_k": pm.ce_acc_anchor,
                    "acc_span_at_k": pm.ce_acc_span,
                    "mrr_anchor": pm.ce_mrr_anchor,
                    "mrr_span": pm.ce_mrr_span,
                    f"ndcg_span@{args.k_max}": pm.ce_ndcg_span,
                }
            paper_entry["per_question"] = [
                {
                    "qid": qr.qid,
                    "question": qr.question,
                    "gold_idxs": qr.gold_idxs,
                    "dense_rank_anchor": qr.dense_rank_anchor,
                    "dense_rank_span": qr.dense_rank_span,
                    "ce_rank_anchor": qr.ce_rank_anchor,
                    "ce_rank_span": qr.ce_rank_span,
                }
                for qr in qrs
            ]
            model_json["papers"].append(paper_entry)

        agg_entry: Dict[str, Any] = {
            "n_questions": agg.n_questions,
            "dense": {
                "acc_anchor_at_k": agg.acc_anchor,
                "acc_span_at_k": agg.acc_span,
                "mrr_anchor": agg.mrr_anchor,
                "mrr_span": agg.mrr_span,
                f"ndcg_span@{args.k_max}": agg.ndcg_span,
            },
        }
        if agg.ce_acc_span is not None:
            agg_entry["ce"] = {
                "acc_anchor_at_k": agg.ce_acc_anchor,
                "acc_span_at_k": agg.ce_acc_span,
                "mrr_anchor": agg.ce_mrr_anchor,
                "mrr_span": agg.ce_mrr_span,
                f"ndcg_span@{args.k_max}": agg.ce_ndcg_span,
            }
        model_json["aggregate"] = agg_entry
        json_output["models"].append(model_json)

    #writing outputs
    if args.out_csv and all_csv_rows:
        write_csv(args.out_csv, all_csv_rows, args.k_max)

    if args.out_json and json_output["models"]:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=2)
        print(f"[JSON] {args.out_json}")

    print("\n>>> Evaluation complete.")


if __name__ == "__main__":
    main()
