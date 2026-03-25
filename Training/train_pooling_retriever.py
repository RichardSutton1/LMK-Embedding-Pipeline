"""train_pooling_retriever.py — generalised pooling bi-encoder fine-tuning.

Extends the LMK training pipeline to support CLS and mean pooling modes,
enabling a fair comparison between pooling strategies when all three have
been trained under identical conditions (same epochs, lr, negatives, etc.).

Pooling modes:
  lmk  — hidden state at last <LMK> token
  cls  — hidden state at [CLS] position
  mean — attention-masked mean of all tokens (sentence-transformers baseline)

Key difference from train_lmk_retriever.py:
  • Accepts --pooling_mode {lmk,cls,mean}.
  • <LMK> special token is only added for pooling_mode=lmk.
  • Window texts use <LMK> separators for lmk, plain space for cls/mean.
  • Query format:  "Question: {q} <LMK>"  for lmk
                   "Question: {q}"         for cls / mean
  • All other hyperparameters, loss, optimiser, and data-loading logic are
    identical to train_lmk_retriever.py so comparisons are fair.


Example commands:
# CLS pooling, full finetune, 16 epochs:
python train_pooling_retriever.py \\
  --model_name sentence-transformers/all-MiniLM-L6-v2 \\
  --pooling_mode cls \\
  --paper_range 14-24 \\
  --use_inbatch --use_mined \\
  --train_mode all --epochs 16 \\
  --out_dir ft_runs/POOLING_CLS_16ep

# Mean pooling:
python train_pooling_retriever.py \\
  --model_name sentence-transformers/all-MiniLM-L6-v2 \\
  --pooling_mode mean \\
  --paper_range 14-24 \\
  --use_inbatch --use_mined \\
  --train_mode all --epochs 16 \\
  --out_dir ft_runs/POOLING_MEAN_16ep

# LMK pooling (identical to train_lmk_retriever.py):
python train_pooling_retriever.py \\
  --model_name sentence-transformers/all-MiniLM-L6-v2 \\
  --pooling_mode lmk \\
  --paper_range 14-24 \\
  --use_inbatch --use_mined \\
  --train_mode all --epochs 16 \\
  --out_dir ft_runs/POOLING_LMK_16ep
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import lmk_train_data as D

# helpers from train_lmk_retriever
from train_lmk_retriever import (
    multi_positive_infonce,
    set_trainable_params,
    TrainExample,
    PaperBundle,
    GlobalExample,
    sample_candidates_for_query,
)


#-----------------------------------------------
# ----------Generalised bi-encoder--------------
#-----------------------------------------------

class PoolingBiEncoder(nn.Module):
    """Shared query/document encoder supporting lmk, cls, and mean pooling"""

    POOLING_MODES = ("lmk", "cls", "mean")

    def __init__(
        self,
        model_name: str,
        pooling_mode: str = "lmk",
        max_length: int = 512,
        add_proj: bool = False,
        proj_dim: int = 256,
    ) -> None:
        super().__init__()
        if pooling_mode not in self.POOLING_MODES:
            raise ValueError(
                f"pooling_mode must be one of {self.POOLING_MODES}, got {pooling_mode!r}"
            )
        self.pooling_mode = pooling_mode

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.encoder   = AutoModel.from_pretrained(model_name)

        # LMK token is only needed for lmk pooling
        if pooling_mode == "lmk":
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<LMK>"]})
            self.encoder.resize_token_embeddings(len(self.tokenizer))
            #left-truncation keeps lmk
            self.tokenizer.truncation_side = "left"
            self.lmk_id: Optional[int] = int(
                self.tokenizer.convert_tokens_to_ids("<LMK>")
            )
        else:
            self.lmk_id = None

        self.tokenizer.padding_side = "right"
        self.max_length = int(max_length)

        hidden = int(self.encoder.config.hidden_size)
        self.proj: Optional[nn.Linear] = (
            nn.Linear(hidden, proj_dim, bias=False) if add_proj else None
        )

    # ------------------------------------------------------------------
    # Query / document formatting
    # ------------------------------------------------------------------

    def format_query(self, question: str) -> str:
        """Format a question string for this encoder's pooling mode"""
        if self.pooling_mode == "lmk":
            return f"Question: {question} <LMK>"
        return f"Question: {question}"

    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------

    def _pool_single(
        self,
        hidden: torch.Tensor,# [T, H]
        input_ids: torch.Tensor, # [T]
        attention_mask: torch.Tensor, #[T]
    ) -> torch.Tensor:
        """extract vector from token hidden states"""
        if self.pooling_mode == "lmk":
            lmk_pos = (input_ids == self.lmk_id).nonzero(as_tuple=False).flatten()
            idx = int(lmk_pos[-1].item()) if lmk_pos.numel() > 0 else 0
            return hidden[idx, :]
        elif self.pooling_mode == "cls":
            return hidden[0, :]
        else:  # mean
            mask = attention_mask.float().unsqueeze(-1)  #[T, 1]
            return (hidden * mask).sum(0) / (mask.sum() + 1e-12)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_texts(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """encode a list of texts -> L2-normalised (B, D) tensor"""
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)

        hs = self.encoder(**enc).last_hidden_state  #[B, T, H]

        vecs: List[torch.Tensor] = []
        for b in range(hs.size(0)):
            v = self._pool_single(hs[b], enc["input_ids"][b], enc["attention_mask"][b])
            vecs.append(v)

        V = torch.stack(vecs, dim=0)  #[B, H]
        if self.proj is not None:
            V = self.proj(V)
        return F.normalize(V, p=2, dim=-1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_checkpoint(self, directory: str) -> None:
        """Save encoder + tokenizer (+ optional projection head) to dir"""
        os.makedirs(directory, exist_ok=True)
        self.encoder.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        if self.proj is not None:
            torch.save(self.proj.state_dict(), os.path.join(directory, "proj_head.pt"))
            print(f"  [saved] proj_head.pt  ({self.proj.weight.shape})")
        #record pooling mode for indetfication
        with open(os.path.join(directory, "pooling_mode.txt"), "w") as f:
            f.write(self.pooling_mode + "\n")


# ----------------------------------------------
# ---------------Data loading-------------------
# -----------------------------------------------

def load_paper_bundle(
    *,
    doc_id: str,
    doc_path: str,
    qa_module: str,
    qa_var: str,
    model_tokenizer: AutoTokenizer,
    model_name: str,
    pooling_mode: str,
    max_length: int,
    window_size: int,
    mined_json: Optional[str],
) -> PaperBundle:
    """Load one paper's text, QA pairs, and mined negatives 

    Mirrors train_lmk_retriever.load_paper_bundle but passes
    use_lmk=(pooling_mode == 'lmk') to build_doc_windows so that
    CLS/mean models receive plain-text windows without <LMK> separators.
    """
    text   = D.load_text_file(doc_path)
    qa_raw = D.load_qa_raw(qa_module, qa_var)

    doc_index = D.build_doc_windows(
        doc_id=doc_id,
        raw_text=text,
        tokenizer=model_tokenizer,
        max_length=max_length,
        window_size=window_size,
        landmark_token="<LMK>",
        use_lmk=(pooling_mode == "lmk"),
    )

    qa: List[TrainExample] = []
    for item in qa_raw:
        qid   = str(item.get("id", "")).strip()
        q     = str(item.get("question", "")).strip()
        golds = D.gold_to_ints(item.get("gold_sentence_idx", None))
        if not qid or not q or not golds:
            continue
        qa.append(TrainExample(qid=qid, question=q, gold_sentence_idx=golds[0]))

    mined = (
        D.load_mined_negatives(mined_json, expected_model=model_name)
        if mined_json else None
    )

    # print(f" loaded {doc_id}: {len(qa)} QA pairs, use_lmk={pooling_mode == 'lmk'}")
    return PaperBundle(
        doc_id=doc_id, doc_path=doc_path,
        qa_module=qa_module, qa_var=qa_var,
        doc_index=doc_index, qa=qa, mined=mined,
    )


# --------------------------------------------
# ----------Training loop---------------------
# -------------------------------------------

def run_training(
    model: PoolingBiEncoder,
    bundles: List[PaperBundle],
    global_qa: List[GlobalExample],
    device: torch.device,
    *,
    out_dir: str,
    epochs: int,
    batch_queries: int,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    grad_accum: int,
    temperature: float,
    use_inbatch: bool,
    use_mined: bool,
    use_random_negs: bool,
    num_random_negs: int,
    num_pos_cap: int,
    num_mined_cap: int,
    pos_mode: str,
    span_pre: int,
    span_post: int,
) -> None:
    """identical to train_lmk_retriever.run_training except
    that query strings are generated via model.format_query() rather than
    being hard-coded to the LMK format
    """
    params = [p for p in model.parameters() if p.requires_grad]
    opt    = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    steps_per_epoch = math.ceil(len(global_qa) / max(1, batch_queries) / max(1, grad_accum))
    total_steps  = max(1, steps_per_epoch * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    os.makedirs(out_dir, exist_ok=True)
    print(f"pooling={model.pooling_mode} temp={temperature:.4g}  "
          f"steps={total_steps}  warmup={warmup_steps}")

    model.train()
    global_step    = 0
    steps_since_log = 0

    for epoch in range(epochs):
        random.shuffle(global_qa)
        running_loss = 0.0
        opt.zero_grad(set_to_none=True)

        for bi in range(0, len(global_qa), batch_queries):
            batch_ex = global_qa[bi : bi + batch_queries]

            batch_q_texts: List[str] = []
            batch_cand_texts: List[List[str]] = []
            batch_pos_masks: List[List[bool]] = []

            for gx in batch_ex:
                b  = bundles[gx.bundle_idx]
                ex = TrainExample(
                    qid=gx.qid,
                    question=gx.question,
                    gold_sentence_idx=gx.gold_sentence_idx,
                )
                pos, negs = sample_candidates_for_query(
                    b, ex,
                    pos_mode=pos_mode,
                    span_pre=span_pre, span_post=span_post,
                    use_mined=use_mined,
                    use_random_negs=use_random_negs,
                    num_random_negs=num_random_negs,
                    num_pos_cap=num_pos_cap,
                    num_mined_cap=num_mined_cap,
                )
                if not pos:
                    continue

                cand_idx   = pos + negs
                cand_texts = [b.doc_index.window_texts[i] for i in cand_idx]
                pos_mask   = [True] * len(pos) + [False] * len(negs)

                #use format_query instead of hard-coded LMK format.
                batch_q_texts.append(model.format_query(ex.question))
                batch_cand_texts.append(cand_texts)
                batch_pos_masks.append(pos_mask)

            if not batch_q_texts:
                continue

            q_vec = model.encode_texts(batch_q_texts, device=device)  #[B, D]

            if use_inbatch:
                flat_cands: List[str] = []
                offsets: List[Tuple[int, int]] = []
                for cands in batch_cand_texts:
                    s = len(flat_cands)
                    flat_cands.extend(cands)
                    offsets.append((s, len(flat_cands)))

                d_vec = model.encode_texts(flat_cands, device=device)  #[K, D]
                sim   = q_vec @ d_vec.T #[B, K]

                pos_mask_t = torch.zeros(
                    (len(batch_q_texts), len(flat_cands)),
                    dtype=torch.bool, device=device,
                )
                for i, (s, e) in enumerate(offsets):
                    for j, is_pos in enumerate(batch_pos_masks[i]):
                        if is_pos:
                            pos_mask_t[i, s + j] = True

                loss = multi_positive_infonce(sim, pos_mask_t, temperature=temperature)
            else:
                losses: List[torch.Tensor] = []
                for i in range(len(batch_q_texts)):
                    d_vec_i = model.encode_texts(batch_cand_texts[i], device=device)
                    sim_i   = q_vec[i : i + 1] @ d_vec_i.T
                    pm_i    = torch.tensor(
                        [batch_pos_masks[i]], dtype=torch.bool, device=device
                    )
                    losses.append(
                        multi_positive_infonce(sim_i, pm_i, temperature=temperature)
                    )
                loss = torch.stack(losses).mean()

            (loss / max(1, grad_accum)).backward()
            running_loss   += float(loss.detach().cpu())

            batch_step = (bi // max(1, batch_queries)) + 1
            if batch_step % max(1, grad_accum) == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                steps_since_log += 1

                if global_step % 20 == 0:
                    avg = running_loss / steps_since_log
                    print(
                        f"  epoch={epoch+1} step={global_step}/{total_steps} "
                        f"loss={avg:.4f}"
                    )
                    running_loss    = 0.0
                    steps_since_log = 0

        #flush accumulated gradients at end of epoch
        if grad_accum > 1:
            remaining = (len(global_qa) // max(1, batch_queries)) % max(1, grad_accum)
            if remaining != 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

        #save per-epoch checkpoint
        ckpt_dir = os.path.join(out_dir, f"epoch_{epoch + 1}")
        model.save_checkpoint(ckpt_dir)
        print(f"  [checkpoint] {ckpt_dir}")


# -----------------
# -----MAIN--------
# -----------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    #Pooling mode
    ap.add_argument(
        "--pooling_mode", type=str, default="lmk",
        choices=PoolingBiEncoder.POOLING_MODES,
        help="Encoder pooling strategy.  'lmk' = LMK token; 'cls' = [CLS]; "
             "'mean' = attention-masked mean.",
    )

    #Paper selection
    ap.add_argument("--manifest",      type=str, default="")
    ap.add_argument("--paper_range",   type=str, default="")
    ap.add_argument("--paper_exclude", type=str, default="")
    ap.add_argument("--doc_prefix",    type=str, default="TEST_PAPER")
    ap.add_argument("--doc_ext",       type=str, default=".mmd")
    ap.add_argument("--qa_prefix",     type=str, default="QA_RAW")
    ap.add_argument("--qa_var",        type=str, default="QA_RAW")
    ap.add_argument("--mined_mode",    type=str, default="auto",
                    choices=["auto", "none", "require"])
    ap.add_argument("--mined_prefix",  type=str, default="mined_TEST_PAPER")
    ap.add_argument("--mined_ext",     type=str, default=".json")

    # Single-paper fallback
    ap.add_argument("--doc_path",   type=str, default="")
    ap.add_argument("--doc_id",     type=str, default="DOC")
    ap.add_argument("--qa_module",  type=str, default="")
    ap.add_argument("--mined_json", type=str, default="")

    #Model / output 
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--out_dir",    type=str, default="lmk_ft_out")

    #window / positives
    ap.add_argument("--window_size", type=int,  default=10)
    ap.add_argument("--max_length",  type=int,  default=512)
    ap.add_argument("--pos_mode",    type=str,  default="span",
                    choices=["span", "exact"])
    ap.add_argument("--span_pre",    type=int,  default=2)
    ap.add_argument("--span_post",   type=int,  default=1)

    # negatives
    ap.add_argument("--use_inbatch",     action="store_true")
    ap.add_argument("--use_mined",       action="store_true")
    ap.add_argument("--use_random_negs", action="store_true")
    ap.add_argument("--num_random_negs", type=int, default=16)
    ap.add_argument("--num_mined_cap",   type=int, default=16)
    ap.add_argument("--num_pos_cap",     type=int, default=4)

    #Training hyperprameters
    ap.add_argument("--device",       type=str,   default="cpu")
    ap.add_argument("--epochs",       type=int,   default=16)
    ap.add_argument("--batch_queries",type=int,   default=8)
    ap.add_argument("--lr",           type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--grad_accum",   type=int,   default=1)
    ap.add_argument("--temperature",  type=float, default=0.05)
    ap.add_argument(
        "--temperature_list", type=str, default="",
        help="Comma-separated temperature sweep, e.g. '0.01,0.05,0.1'.",
    )
    ap.add_argument("--seed", type=int, default=42)

    # Freezing mode
    ap.add_argument("--train_mode", type=str, default="last_n",
                    choices=["lmk_only", "last_n", "all"],
                    help="'lmk_only' trains token embeddings only (note: for "
                         "cls/mean pooling this trains base embeddings with no "
                         "new <LMK> row); 'last_n' unfreezes last N layers; "
                         "'all' full fine-tune.")
    ap.add_argument("--train_last_n_layers", type=int, default=2)

    # (optional)
    ap.add_argument("--add_proj",  action="store_true")
    ap.add_argument("--proj_dim",  type=int, default=256)

    args = ap.parse_args()

    #warn if lmk_only mode is used with non-LMK pooling
    if args.train_mode == "lmk_only" and args.pooling_mode != "lmk":
        print(
            f"[WARNING] --train_mode lmk_only with --pooling_mode {args.pooling_mode}: "
            "will train token embeddings only (no <LMK> row exists for this mode). "
            "Consider --train_mode last_n or all for CLS/mean pooling."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    #Paper rows
    rows: List[Dict[str, Any]] = []
    if args.manifest:
        if not os.path.exists(args.manifest):
            raise FileNotFoundError(args.manifest)
        with open(args.manifest, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif args.paper_range:
        include = D.parse_int_range(args.paper_range)
        exclude = set(D.parse_int_range(args.paper_exclude))
        idxs    = [i for i in include if i not in exclude]
        if not idxs:
            raise ValueError("--paper_range produced no papers after exclude filter.")
        rows = D.build_rows_from_paper_indices(
            idxs,
            doc_prefix=args.doc_prefix, doc_ext=args.doc_ext,
            qa_prefix=args.qa_prefix,   qa_var=args.qa_var,
            mined_prefix=args.mined_prefix, mined_ext=args.mined_ext,
            mined_mode=args.mined_mode,
        )
    else:
        if not args.doc_path or not args.qa_module:
            raise ValueError(
                "Provide --manifest, --paper_range, or (--doc_path and --qa_module)."
            )
        rows = [{
            "doc_id": args.doc_id, "doc_path": args.doc_path,
            "qa_module": args.qa_module, "qa_var": args.qa_var,
            "mined_json": args.mined_json or "",
        }]

    #resolve temperature list
    if args.temperature_list.strip():
        temperatures = [float(t.strip()) for t in args.temperature_list.split(",")]
    else:
        temperatures = [args.temperature]

    #build a temporary model just to get the tokeniser
    _tmp = PoolingBiEncoder(
        model_name=args.model_name,
        pooling_mode=args.pooling_mode,
        max_length=args.max_length,
        add_proj=args.add_proj,
        proj_dim=args.proj_dim,
    )

    # load paper bundles (shared for all temp)
    bundles: List[PaperBundle] = []
    for r in rows:
        doc_path = str(r["doc_path"])
        if not os.path.exists(doc_path):
            raise FileNotFoundError(f"Missing: {doc_path}")
        b = load_paper_bundle(
            doc_id=str(r.get("doc_id", "DOC")),
            doc_path=doc_path,
            qa_module=str(r["qa_module"]),
            qa_var=str(r.get("qa_var", "QA_RAW")),
            model_tokenizer=_tmp.tokenizer,
            model_name=args.model_name,
            pooling_mode=args.pooling_mode,
            max_length=args.max_length,
            window_size=args.window_size,
            mined_json=str(r.get("mined_json", "") or "") or None,
        )
        bundles.append(b)

    global_qa: List[GlobalExample] = [
        GlobalExample(
            bundle_idx=bi,
            qid=ex.qid,
            question=ex.question,
            gold_sentence_idx=ex.gold_sentence_idx,
        )
        for bi, b in enumerate(bundles)
        for ex in b.qa
    ]

    if not global_qa:
        raise RuntimeError("No QA examples loaded.")

    print(f">>> Loaded {len(bundles)} papers, {len(global_qa)} total QA examples")
    print(f">>> pooling_mode={args.pooling_mode}  train_mode={args.train_mode}  "
          f"last_n={args.train_last_n_layers}")
    print(f">>> Negatives: inbatch={args.use_inbatch}  mined={args.use_mined}  "
          f"random={args.use_random_negs}")
    print(f">>> Temperature sweep: {temperatures}")

    #Temp sweep
    for temp in temperatures:
        print(f"\n{'='*60}")
        print(f"  POOLING={args.pooling_mode.upper()}  TEMPERATURE={temp}")
        print(f"{'='*60}")

        model = PoolingBiEncoder(
            model_name=args.model_name,
            pooling_mode=args.pooling_mode,
            max_length=args.max_length,
            add_proj=args.add_proj,
            proj_dim=args.proj_dim,
        ).to(device)
        set_trainable_params(
            model, mode=args.train_mode,
            last_n_layers=args.train_last_n_layers,
        )

        temp_dir = (
            os.path.join(args.out_dir, f"temp_{temp:.4g}")
            if len(temperatures) > 1
            else args.out_dir
        )

        run_training(
            model=model,
            bundles=bundles,
            global_qa=global_qa,
            device=device,
            out_dir=temp_dir,
            epochs=args.epochs,
            batch_queries=args.batch_queries,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            grad_accum=args.grad_accum,
            temperature=temp,
            use_inbatch=args.use_inbatch,
            use_mined=args.use_mined,
            use_random_negs=args.use_random_negs,
            num_random_negs=args.num_random_negs,
            num_pos_cap=args.num_pos_cap,
            num_mined_cap=args.num_mined_cap,
            pos_mode=args.pos_mode,
            span_pre=args.span_pre,
            span_post=args.span_post,
        )

    print("\n>>> All done.")


if __name__ == "__main__":
    main()
