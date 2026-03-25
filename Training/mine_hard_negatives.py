"""mine_hard_negatives.py — mine hard negatives for contrastive training

For each QA item: embed the question and all document anchors, then collect
the highest-scoring non-positive anchors as hard negatives.



# single paper:
python mine_hard_negatives.py \\
    --doc_path TEST_PAPER1.mmd --doc_id TEST_PAPER1 \\
    --qa_module QA_RAW1 \\
    --model_name lmk_finetuned_epoch2 \\
    --out_json mined_TEST_PAPER1.json \\
    --pos_mode span --span_pre 2 --span_post 1 \\
    --window_size 10 --dense_top_n 200 --num_hard_negs 20

# all papers in range:
python mine_hard_negatives.py \\
    --paper_range 1-24 \\
    --model_name lmk_finetuned_epoch2
"""
from __future__ import annotations
import argparse
from typing import Any, Dict, List
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import lmk_train_data as D



# --- LMK encoder ---

class LMKEncoder:
    """Lightweight LMK encoder for mining -- no projection head, no gradients."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<LMK>"]})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "right"

        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.lmk_id = int(self.tokenizer.convert_tokens_to_ids("<LMK>"))
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)

    @torch.no_grad()
    def embed_last_lmk(self, texts: List[str]) -> np.ndarray:
        """embeds text using the last LMK hidden state"""
        vecs: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            hs = self.model(**enc).last_hidden_state
            input_ids = enc["input_ids"]
            for b in range(input_ids.size(0)):
                lmk_pos = (input_ids[b] == self.lmk_id).nonzero(as_tuple=False).flatten()
                idx = int(lmk_pos[-1].item()) if lmk_pos.numel() > 0 else 0
                v = hs[b, idx, :].float().cpu().numpy()
                v = v / (np.linalg.norm(v) + 1e-12)
                vecs.append(v)
        return np.stack(vecs, axis=0)


def cosine_scores(query_vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    q = query_vec.astype(np.float32)
    M = mat.astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-9)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return M @ q




# --- mining logic for a single paper ---

def mine_paper(
    *,
    doc_id: str,
    doc_path: str,
    qa_module: str,
    qa_var: str,
    out_json: str,
    encoder: LMKEncoder,
    model_name: str,
    pos_mode: str,
    span_pre: int,
    span_post: int,
    window_size: int,
    dense_top_n: int,
    num_hard_negs: int,
) -> None:
    text = D.load_text_file(doc_path)
    qa = D.load_qa_raw(qa_module, qa_var)

    doc_idx = D.build_doc_windows(
        doc_id=doc_id,
        raw_text=text,
        tokenizer=encoder.tokenizer,
        max_length=encoder.max_length,
        window_size=window_size,
        landmark_token="<LMK>",
        use_lmk=True,
    )

    if not doc_idx.sentences:
        raise RuntimeError(f"No sentences found in {doc_path}")

    print(f"  Building anchor embeddings for {len(doc_idx.window_texts)} anchors ...")
    anchor_vecs = encoder.embed_last_lmk(doc_idx.window_texts)  # [N, H]

    mined: Dict[str, Any] = {
        "doc_id": doc_id,
        "doc_path": doc_path,
        "qa_module": qa_module,
        "model_name": model_name,
        "pos_mode": pos_mode,
        "span_pre": span_pre,
        "span_post": span_post,
        "window_size": window_size,
        "max_length": encoder.max_length,
        "items": {},
    }

    for item in qa:
        qid = str(item.get("id", "")).strip()
        q = str(item.get("question", "")).strip()
        golds = D.gold_to_ints(item.get("gold_sentence_idx", None))

        if not qid or not q or not golds:
            continue

        n_sent = len(doc_idx.sentences)
        pos_set: set = set()
        for g in golds:
            pos_set.update(
                D.positive_anchor_set(g, n_sent, mode=pos_mode,
                                      span_pre=span_pre, span_post=span_post)
            )

        q_text = f"Question: {q} <LMK>"
        q_vec = encoder.embed_last_lmk([q_text])[0]
        scores = cosine_scores(q_vec, anchor_vecs)
        top_idx = np.argsort(-scores)[: min(dense_top_n, len(scores))]

        #collect hard negs
        hard_negs: List[int] = []
        for i in top_idx.tolist():
            if i in pos_set:
                continue
            hard_negs.append(int(i))
            if len(hard_negs) >= num_hard_negs:
                break

        # print(f"  {qid}: {len(hard_negs)} hard negs found")

        mined["items"][qid] = {
            "question": q,
            "gold_sentence_idx": golds[0] if len(golds) == 1 else golds,
            "pos_anchors": sorted(pos_set),
            "hard_neg_anchors": hard_negs,
        }

    D.save_mined_negatives(out_json, mined)
    print(f"  Saved -> {out_json}  ({len(mined['items'])} QAs)")






#--------------
# -----MAIN----
#--------------

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #papers
    ap.add_argument("--paper_range", type=str, default="",
                    help="Paper index range, e.g. '1-24'. Auto-names output files.")
    ap.add_argument("--paper_exclude", type=str, default="",
                    help="Indices to exclude from --paper_range.")
    ap.add_argument("--doc_prefix", type=str, default="TEST_PAPER")
    ap.add_argument("--doc_ext", type=str, default=".mmd")
    ap.add_argument("--qa_prefix", type=str, default="QA_RAW")
    ap.add_argument("--qa_var", type=str, default="QA_RAW")
    ap.add_argument("--mined_prefix", type=str, default="mined_TEST_PAPER",
                    help="Output file prefix when using --paper_range.")
    ap.add_argument("--mined_ext", type=str, default=".json")

    #fallback
    ap.add_argument("--doc_path", type=str, default="")
    ap.add_argument("--doc_id", type=str, default="DOC")
    ap.add_argument("--qa_module", type=str, default="")
    ap.add_argument("--out_json", type=str, default="",
                    help="Output file (required for single-paper mode).")

    #model
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)

    #mining hyperparams
    ap.add_argument("--window_size", type=int, default=10)
    ap.add_argument("--pos_mode", type=str, default="span", choices=["span", "exact"])
    ap.add_argument("--span_pre", type=int, default=2)
    ap.add_argument("--span_post", type=int, default=1)
    ap.add_argument("--dense_top_n", type=int, default=200)
    ap.add_argument("--num_hard_negs", type=int, default=20)

    args = ap.parse_args()

    encoder = LMKEncoder(
        model_name=args.model_name,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    #Resolve paper list
    if args.paper_range:
        include = D.parse_int_range(args.paper_range)
        exclude = set(D.parse_int_range(args.paper_exclude))
        idxs = [i for i in include if i not in exclude]
        if not idxs:
            raise ValueError("--paper_range produced no papers after exclude filter.")

        rows = D.build_rows_from_paper_indices(
            idxs,
            doc_prefix=args.doc_prefix, doc_ext=args.doc_ext,
            qa_prefix=args.qa_prefix, qa_var=args.qa_var,
            mined_prefix=args.mined_prefix, mined_ext=args.mined_ext,
            mined_mode="none",
        )
        for r, i in zip(rows, idxs):
            r["out_json"] = f"{args.mined_prefix}{i}{args.mined_ext}"
    else:
        if not args.doc_path or not args.qa_module or not args.out_json:
            ap.error("Provide --paper_range OR (--doc_path, --qa_module, --out_json).")
        rows = [{
            "doc_id": args.doc_id,
            "doc_path": args.doc_path,
            "qa_module": args.qa_module,
            "qa_var": args.qa_var,
            "out_json": args.out_json,
        }]

    print(f">>> Mining {len(rows)} paper(s) with model: {args.model_name}")
    for r in rows:
        print(f"\n--- {r['doc_id']} -> {r['out_json']}")
        mine_paper(
            doc_id=r["doc_id"],
            doc_path=r["doc_path"],
            qa_module=r["qa_module"],
            qa_var=r.get("qa_var", "QA_RAW"),
            out_json=r["out_json"],
            encoder=encoder,
            model_name=args.model_name,
            pos_mode=args.pos_mode,
            span_pre=args.span_pre,
            span_post=args.span_post,
            window_size=args.window_size,
            dense_top_n=args.dense_top_n,
            num_hard_negs=args.num_hard_negs,
        )

    print("\n>>> Done.")


if __name__ == "__main__":
    main()
