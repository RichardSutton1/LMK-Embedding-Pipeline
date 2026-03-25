"""lmk_train_data.py — shared data utilities for the LMK pipeline.

Sentence splitting, reference removal, window construction (with/without <LMK>),
DocIndex, positive anchor sets, paper index helpers, mined-negative I/O.
"""
from __future__ import annotations
import importlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch
from transformers import AutoTokenizer


# --- sentence splitting ---

def split_into_sentences(text: str) -> List[str]:
    """Split on .!? followed by whitespace.

    NOTE: keep this the same regex as the original pipeline so sentence
    indices stay aligned with existing gold_sentence_idx annotations.
    Don't add abbreviation handling here without re-annotating QA files.
    """
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


# --- reference removal ---

def remove_references(text: str) -> str:
    """Truncate at first References / Acknowledgements / Appendix heading.

    Tries markdown headers first; falls back to bare-word match (only if
    it appears after char 1500, to avoid false positives in abstracts).
    """
    header_patterns = (
        r"(?im)^\s*#{1,6}\s*References\s*$",
        r"(?im)^\s*#{1,6}\s*Acknowledg(?:e)?ments\s*$",
        r"(?im)^\s*#{1,6}\s*Appendix\b.*$",
    )
    cut_positions: List[int] = []
    for pat in header_patterns:
        m = re.search(pat, text)
        if m:
            cut_positions.append(m.start())
    if cut_positions:
        return text[: min(cut_positions)]

    # fallback: bare word, guareded by position to avoid picking up
    # stray "References" mentions in the abstract
    m = re.search(r"\b(References|Acknowledg(?:e)?ments|Appendix)\b", text, flags=re.I)
    if m and m.start() > 1500:
        return text[: m.start()]

    return text


# --- loaders ---

def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_qa_raw(qa_module: str, qa_var: str = "QA_RAW") -> List[Dict[str, Any]]:
    mod = importlib.import_module(qa_module)
    if not hasattr(mod, qa_var):
        raise AttributeError(
            f"Module '{qa_module}' missing '{qa_var}'. "
            f"Available: {[a for a in dir(mod) if not a.startswith('_')]}"
        )
    qa = getattr(mod, qa_var)
    if not isinstance(qa, list):
        raise TypeError(f"{qa_module}.{qa_var} must be a list, got {type(qa)}")
    return qa


def gold_to_ints(g: Any) -> Optional[List[int]]:
    """Normalise gold_sentence_idx to List[int] (handles int, float, list)
    Returns None for invalid/missing values
    """
    if g is None:
        return None
    if isinstance(g, list):
        try:
            return [int(x) for x in g]
        except (TypeError, ValueError):
            return None
    try:
        return [int(g)]
    except (TypeError, ValueError):
        return None


# --- window helpers ---

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def window_text_with_lmks(sents: Sequence[str], lmk: str = "<LMK>") -> str:
    """Join sentences with <LMK> separators: 'sent1 <LMK> sent2 <LMK>'"""
    parts: List[str] = []
    for s in sents:
        parts.append(s)
        parts.append(lmk)
    return " ".join(parts)


def window_text_plain(sents: Sequence[str]) -> str:
    """Plain join for CLS/mean-pooling models (no LMK tokens)."""
    return " ".join(sents)


def fits_without_truncation(tokenizer: AutoTokenizer, text: str, max_len: int) -> bool:
    ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    return len(ids) <= max_len


# --- DocIndex + window builder ---

@dataclass
class DocIndex:
    doc_id: str
    sentences: List[str]
    window_texts: List[str] #one window per anchor sentence
    kept_context_counts: List[int] #how many sentences ended up in the window


def build_doc_windows(
    *,
    doc_id: str,
    raw_text: str,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    window_size: int = 10,
    landmark_token: str = "<LMK>",
    use_lmk: bool = True,
) -> DocIndex:
    """Builds one context window per sentence anchor"""
    clean = remove_references(raw_text)
    sentences = split_into_sentences(clean)

    window_texts: List[str] = []
    kept_counts: List[int] = []

    for end_idx in range(len(sentences)):
        start_idx = max(0, end_idx - window_size + 1)
        window_sents = sentences[start_idx : end_idx + 1]

        #shrink from left
        while True:
            w_text = (
                window_text_with_lmks(window_sents, landmark_token)
                if use_lmk
                else window_text_plain(window_sents)
            )
            if fits_without_truncation(tokenizer, w_text, max_length):
                break
            if len(window_sents) <= 1:
                break
            window_sents = window_sents[1:]

        #print(f"window for sent {end_idx}: {len(window_sents)} sents kept")

        w_text = (
            window_text_with_lmks(window_sents, landmark_token)
            if use_lmk
            else window_text_plain(window_sents)
        )
        window_texts.append(w_text)
        kept_counts.append(len(window_sents))

    return DocIndex(
        doc_id=doc_id,
        sentences=sentences,
        window_texts=window_texts,
        kept_context_counts=kept_counts,
    )


# --- positive anchor set ---

def positive_anchor_set(
    gold_sentence_idx: int,
    n_sent: int,
    *,
    mode: str = "span",
    span_pre: int = 2,
    span_post: int = 1,
) -> List[int]:
    """Return anchor indices that count as positives for a given gold sentence

    mode="exact": only the gold index.
    mode="span":  [gold - span_pre, gold + span_post] inclusive.
                  With defaults (pre=2, post=1) -> [gold-2, gold-1, gold, gold+1].
                  Rationale: anchors near gold tend to produce windows that
                  contain the gold sentence when retrieved.
    """
    g = int(gold_sentence_idx)

    if mode == "exact":
        if 0 <= g < n_sent:
            return [g]
        return []

    if mode != "span":
        raise ValueError("mode must be 'span' or 'exact'")

    lo = clamp(g - span_pre, 0, n_sent - 1)
    hi = clamp(g + span_post, 0, n_sent - 1)
    return list(range(lo, hi + 1))


# ---- paper-index helpers (shared by train + mine scripts) ----

def parse_int_range(spec: str) -> List[int]:

    spec = (spec or "").strip()
    if not spec:
        return []
    out: List[int] = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if "-" in part:
            a_str, b_str = part.split("-", 1)
            a, b = int(a_str.strip()), int(b_str.strip())
            out.extend(range(a, b + 1) if a <= b else range(a, b - 1, -1))
        else:
            out.append(int(part))
    #dedupe, preserve order
    seen: set = set()
    return [x for x in out if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]


def build_rows_from_paper_indices(
    idxs: List[int],
    *,
    doc_prefix: str = "TEST_PAPER",
    doc_ext: str = ".mmd",
    qa_prefix: str = "QA_RAW",
    qa_var: str = "QA_RAW",
    mined_prefix: str = "mined_TEST_PAPER",
    mined_ext: str = ".json",
    mined_mode: str = "auto",  #"auto" | "none" | "require"
) -> List[Dict[str, Any]]:
    """Build a list of paper-config dicts from paper indices"""
    rows: List[Dict[str, Any]] = []
    for i in idxs:
        doc_id = f"{doc_prefix}{i}"
        mined_json = f"{mined_prefix}{i}{mined_ext}"

        if mined_mode == "none":
            mined_json = ""
        elif mined_mode == "auto":
            if not os.path.exists(mined_json):
                mined_json = ""
        elif mined_mode == "require":
            if not os.path.exists(mined_json):
                raise FileNotFoundError(
                    f"mined_mode=require but missing: {mined_json}. "
                    "Generate it first or use --mined_mode auto/none."
                )
        else:
            raise ValueError("mined_mode must be one of: auto, none, require")

        rows.append({
            "doc_id": doc_id,
            "doc_path": f"{doc_id}{doc_ext}",
            "qa_module": f"{qa_prefix}{i}",
            "qa_var": qa_var,
            "mined_json": mined_json,
        })
    return rows


# --- hard-negative I/O ----

def save_mined_negatives(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_mined_negatives(
    path: str,
    expected_model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Load mined-negatives JSON. Warns if the model name doesn't match."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if expected_model and "model_name" in data:
        if data["model_name"] != expected_model:
            print(
                f"[WARNING] Mined negatives in '{path}' were built with "
                f"model '{data['model_name']}', but current model is "
                f"'{expected_model}'. Consider re-mining for best results."
            )
    return data
