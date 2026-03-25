"""LMK_Embed10_DENSE_ONLY.py

LMK embedding RAG pipeline (dense retrieval only).

Encoder class supports three pooling modes so LMK and baseline models
can be evaluated through the same code path:
  pooling_mode="lmk"  — hidden state at last <LMK> token
  pooling_mode="cls"  — hidden state at [CLS] position
  pooling_mode="mean" — mean over non-padding tokens

Pipeline:
  1. Split document into sentences.
  2. Build context window per anchor sentence (with or without <LMK> separators).
  3. Dense retrieval via cosine similarity.
  4. Build spans, optionally rerank with cross-encoder.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

from lmk_train_data import (
    split_into_sentences,
    remove_references,
    load_text_file,
    window_text_with_lmks,
    window_text_plain,
    fits_without_truncation,
)


# --- encoder ---

class Encoder:
    """Universal sentence encoder supporting lmk / cls / mean pooling.

    Args:
        model_name:   HuggingFace model ID or local path.
        pooling_mode: "lmk" | "cls" | "mean"
        device:       torch device string (auto-detected if None).
        batch_size:   encoding batch size.
        max_length:   tokeniser max length.
    """

    POOLING_MODES = ("lmk", "cls", "mean")

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        pooling_mode: str = "lmk",
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        if pooling_mode not in self.POOLING_MODES:
            raise ValueError(
                f"pooling_mode must be one of {self.POOLING_MODES}, got '{pooling_mode}'"
            )
        self.pooling_mode = pooling_mode

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)

        if pooling_mode == "lmk":
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<LMK>"]})
            self.model.resize_token_embeddings(len(self.tokenizer))
            # eft-truncation keeps the anchor <LMK> at the end of long windows
            self.tokenizer.truncation_side = "left"
            self.lmk_id: Optional[int] = int(
                self.tokenizer.convert_tokens_to_ids("<LMK>")
            )
        else:
            self.lmk_id = None

        self.tokenizer.padding_side = "right"

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.vector_size = int(self.model.config.hidden_size)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)

    # --- helpers ----

    def format_query(self, query: str) -> str:
        """Wrap a question string in the format expected by this encoder."""
        if self.pooling_mode == "lmk":
            return f"Question: {query} <LMK>"
        return f"Question: {query}"

    def window_text(self, sents: List[str]) -> str:
        """Build a window text for a list of sentences."""
        if self.pooling_mode == "lmk":
            return window_text_with_lmks(sents)
        return window_text_plain(sents)

    # --- embedding ---

    @torch.no_grad()
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts. Returns (N, H) float32 numpy array, L2-normalised.

        lmk mode: hidden state at last <LMK> position.
        cls mode: [CLS] token.
        mean mode: attention-masked mean over all token positions.
        """
        if isinstance(texts, str):
            texts = [texts]

        # print(f"embedding {len(texts)} texts, mode={self.pooling_mode}")

        all_vecs: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            hs = self.model(**enc).last_hidden_state  #[B, T, H]

            for b in range(hs.size(0)):
                vec = self._pool(hs[b], enc["input_ids"][b], enc["attention_mask"][b])
                all_vecs.append(vec)

        return np.stack(all_vecs, axis=0).astype(np.float32)

    @torch.no_grad()
    def embed_lmk_positions(self, texts: List[str]) -> List[List[List[float]]]:
        """Return embeddings at ALL <LMK> positions per text (lmk mode only).

        Returns a list where each element is a (K_i x H) list of vectors.
        Falls back to [CLS] vector for texts with no <LMK> token.
        Only meaningful when pooling_mode == "lmk".
        """
        if self.pooling_mode != "lmk":
            raise RuntimeError("embed_lmk_positions is only valid for pooling_mode='lmk'")
        if isinstance(texts, str):
            texts = [texts]

        outputs: List[List[List[float]]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            hs = self.model(**enc).last_hidden_state
            input_ids = enc["input_ids"]

            for b in range(hs.size(0)):
                lmk_pos = (input_ids[b] == self.lmk_id).nonzero(as_tuple=False).flatten()
                if lmk_pos.numel() == 0:
                    v = self._pool(hs[b], input_ids[b], enc["attention_mask"][b])
                    outputs.append([v.tolist()])
                else:
                    vecs: List[List[float]] = []
                    for pos in lmk_pos.tolist():
                        v = hs[b, int(pos), :].float().cpu().numpy()
                        v = v / (np.linalg.norm(v) + 1e-12)
                        vecs.append(v.tolist())
                    outputs.append(vecs)

        return outputs



    # --- helpers---
    def _pool(
        self,
        hidden: torch.Tensor, #[T, H]
        input_ids: torch.Tensor, # [T]
        attention_mask: torch.Tensor, #[T]
    ) -> np.ndarray:
        """extracts single vector from token hidden states using the chosen pooling mode."""
        if self.pooling_mode == "lmk":
            lmk_pos = (input_ids == self.lmk_id).nonzero(as_tuple=False).flatten()
            idx = int(lmk_pos[-1].item()) if lmk_pos.numel() > 0 else 0
            vec = hidden[idx, :].float().cpu().numpy()
        elif self.pooling_mode == "cls":
            vec = hidden[0, :].float().cpu().numpy()
        else:  #mean
            mask = attention_mask.unsqueeze(-1).float().cpu()
            h = hidden.float().cpu()
            vec = ((h * mask).sum(0) / (mask.sum() + 1e-12)).numpy()

        return vec / (np.linalg.norm(vec) + 1e-12)




class LMK_Embedding(Encoder):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device=None,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        super().__init__(
            model_name=model_name,
            pooling_mode="lmk",
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )


# --- cross-encoder reranker ---

class CrossEncoderReranker:
    """Score (query, passage) pairs with a cross-encoder. Higher = more relevant."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_pairs(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 16,
        max_length: int = 512,
    ) -> List[float]:
        scores: List[float] = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            qs = [p[0] for p in batch]
            ts = [p[1] for p in batch]
            enc = self.tokenizer(
                qs, ts,
                padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            # single-output model: raw logit is the relevance score
            # two-output model: take the positive class (index 1)
            batch_scores = logits.squeeze(-1) if logits.shape[-1] == 1 else logits[:, 1]
            scores.extend(batch_scores.detach().cpu().tolist())
        return scores




# --- dense index builder ---

def compute_sentence_embeddings_window_fit(
    encoder: Encoder,
    text: str,
    window_size: int = 15,
) -> Dict[str, Any]:
    """Builds dense index (sentence embeddings + metadata) for one document"""
    clean = remove_references(text)
    sentences = split_into_sentences(clean)

    if not sentences:
        return {
            "sentences": [],
            "sentence_embeddings": np.zeros((0, encoder.vector_size), dtype=np.float32),
            "kept_context_sent_count": [],
        }

    n = len(sentences)
    window_texts: List[str] = []
    kept_counts: List[int] = []

    for end_idx in range(n):
        window_sents = sentences[max(0, end_idx - window_size + 1) : end_idx + 1]
        while True:
            w_text = encoder.window_text(window_sents)
            if fits_without_truncation(encoder.tokenizer, w_text, encoder.max_length):
                break
            if len(window_sents) <= 1:
                break
            window_sents = window_sents[1:]
        window_texts.append(encoder.window_text(window_sents))
        kept_counts.append(len(window_sents))

    # embed all windows
    sent_embs: List[Optional[np.ndarray]] = [None] * n
    if encoder.pooling_mode == "lmk":
        # extracting the last LMK vector per window
        for i in range(0, len(window_texts), encoder.batch_size):
            batch_texts = window_texts[i : i + encoder.batch_size]
            batch_vecs = encoder.embed_lmk_positions(batch_texts)
            for j, lmk_vecs in enumerate(batch_vecs):
                if not lmk_vecs:
                    raise RuntimeError(f"No LMK vectors found for sentence index {i + j}")
                sent_embs[i + j] = np.array(lmk_vecs[-1], dtype=np.float32)
    else:
        #CLS / mean: embed all windows directly
        all_vecs = encoder.embed(window_texts)
        for i, v in enumerate(all_vecs):
            sent_embs[i] = v

    missing = [k for k, v in enumerate(sent_embs) if v is None]
    if missing:
        raise RuntimeError(f"Missing embeddings for indices (first 20): {missing[:20]}")

    sent_embs_np = np.stack([v for v in sent_embs if v is not None], axis=0)

    return {
        "sentences": sentences,
        "sentence_embeddings": sent_embs_np,
        "kept_context_sent_count": kept_counts,
    }


def build_sentence_index_dense_only(
    text: str,
    embedding_model: Encoder,
    window_size: int = 10,
) -> Dict[str, Any]:
    return compute_sentence_embeddings_window_fit(
        encoder=embedding_model,
        text=text,
        window_size=window_size,
    )


# ----- dense retrieval -----

def cosine_scores_matrix(query_vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Cosine similarity between query vector and passage vectors"""
    q = query_vec.astype(np.float32)
    M = mat.astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-9)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return M @ q


def retrieve_dense_top_n(
    query: str,
    encoder: Encoder,
    index: Dict[str, Any],
    n: int = 50,
) -> List[Dict[str, Any]]:
    """Retrieves top-n sentence anchors by cosine similarity"""
    sentences: List[str] = index.get("sentences", [])
    sent_embs: np.ndarray = index["sentence_embeddings"]

    if not sentences:
        return []

    q_text = encoder.format_query(query)
    q_vec = encoder.embed([q_text])[0]
    scores = cosine_scores_matrix(q_vec, sent_embs)
    top_idx = np.argsort(-scores)[: min(n, len(sentences))]

    return [
        {"sentence_idx": int(i), "dense_score": float(scores[i])}
        for i in top_idx
    ]


# --- span building + CE reranking ---

@dataclass
class CandidateSpan:
    anchor_i: int
    lo: int
    hi: int
    span_text: str
    dense_score: float = 0.0
    ce_score: Optional[float] = None


def _make_span(sentences: List[str], anchor_i: int, pre: int = 2, post: int = 1):
    lo = max(0, anchor_i - pre)
    hi = min(len(sentences) - 1, anchor_i + post)
    return lo, hi, " ".join(sentences[lo : hi + 1])


def answer_query_dense_ce_span(
    query: str,
    encoder: Encoder,
    index: Dict[str, Any],
    reranker: Optional[CrossEncoderReranker] = None,
    dense_top_n: int = 100,
    ce_batch_size: int = 16,
    span_pre: int = 2,
    span_post: int = 1,
) -> Dict[str, Any]:
    """
    retrieve -> build spans -> CE rerank
    """
    sentences: List[str] = index.get("sentences", [])
    if not sentences:
        return {"best": None, "candidates": [], "dense_hits": []}

    dense_hits = retrieve_dense_top_n(query, encoder, index, n=dense_top_n)
    if not dense_hits:
        return {"best": None, "candidates": [], "dense_hits": []}

    candidates: List[CandidateSpan] = []
    for h in dense_hits:
        i = int(h["sentence_idx"])
        lo, hi, span_text = _make_span(sentences, i, pre=span_pre, post=span_post)
        candidates.append(CandidateSpan(
            anchor_i=i, lo=lo, hi=hi, span_text=span_text,
            dense_score=float(h.get("dense_score", 0.0)),
        ))

    if reranker is None:
        candidates.sort(key=lambda c: c.dense_score, reverse=True)
        best = candidates[0]
    else:
        pairs = [(query, c.span_text) for c in candidates]
        ce_scores = reranker.score_pairs(pairs, batch_size=ce_batch_size)
        for c, s in zip(candidates, ce_scores):
            c.ce_score = float(s)
        candidates.sort(key=lambda c: float(c.ce_score if c.ce_score is not None else -1e9), reverse=True)
        best = candidates[0]

    return {
        "best": {
            "hit_sentence_idx": best.anchor_i,
            "span_lo": best.lo,
            "span_hi": best.hi,
            "answer": best.span_text,
            "dense_score": best.dense_score,
            "ce_score": best.ce_score,
        },
        "candidates": [c.__dict__ for c in candidates],
        "dense_hits": dense_hits,
    }


#----------MAIN-----------

def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive LMK dense-only QA over a single document.")
    ap.add_argument("--doc_path", type=str, default="TEST_PAPER1.mmd")
    ap.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--pooling_mode", type=str, default="lmk", choices=Encoder.POOLING_MODES)
    ap.add_argument("--ce_model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--no_ce", action="store_true")
    ap.add_argument("--window_size", type=int, default=10)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--dense_top_n", type=int, default=100)
    ap.add_argument("--span_pre", type=int, default=2)
    ap.add_argument("--span_post", type=int, default=1)
    ap.add_argument("--ce_batch_size", type=int, default=16)
    args = ap.parse_args()

    text = load_text_file(args.doc_path)
    enc = Encoder(
        model_name=args.model_name,
        pooling_mode=args.pooling_mode,
        batch_size=16,
        max_length=args.max_length,
    )
    ce = None if args.no_ce else CrossEncoderReranker(model_name=args.ce_model)

    index = build_sentence_index_dense_only(text, enc, window_size=args.window_size)
    print(
        f">>> Indexed {len(index.get('sentences', []))} sentences "
        f"(mode={args.pooling_mode}, dense_only)"
    )

    while True:
        q = input("Question> ").strip()
        if q.lower() in {"exit", "quit", "q"}:
            break
        out = answer_query_dense_ce_span(
            query=q, encoder=enc, index=index, reranker=ce,
            dense_top_n=args.dense_top_n, ce_batch_size=args.ce_batch_size,
            span_pre=args.span_pre, span_post=args.span_post,
        )
        best = out.get("best")
        if not best:
            print("No answer.\n")
            continue
        print(f"\n>>> ANSWER (anchor={best['hit_sentence_idx']}, "
              f"span={best['span_lo']}-{best['span_hi']}, "
              f"dense={best['dense_score']:.4f}, ce={best['ce_score']})")
        print(best["answer"], "\n")


if __name__ == "__main__":
    main()
