"""train_lmk_retriever.py — contrastive bi-encoder fine-tuning for LMK retrieval.

Trains a shared query/document encoder using multi-positive InfoNCE loss.
Supports training over multiple papers in one run (via --paper_range or
--manifest), optional hard negatives from pre-mined JSON files, and a
temperature sweep that saves separate checkpoints per temperature.

Example commands
----------------
#Single paper, LMK-embedding-only training:
python train_lmk_retriever.py \\
  --doc_path TEST_PAPER1.mmd --doc_id TEST_PAPER1 \\
  --qa_module QA_RAW1 \\
  --model_name sentence-transformers/all-MiniLM-L6-v2 \\
  --out_dir ft_runs/runA \\
  --use_inbatch \\
  --train_mode lmk_only \\
  --pos_mode span --span_pre 2 --span_post 1 \\
  --epochs 3 --batch_queries 6 --lr 5e-5

#multiple papers, mined negatives, last-2-layer finetune, temperature sweep:
python train_lmk_retriever.py \\
  --paper_range 1-24 \\
  --model_name sentence-transformers/all-MiniLM-L6-v2 \\
  --out_dir ft_runs/sweep \\
  --use_inbatch --use_mined \\
  --train_mode last_n --train_last_n_layers 2 \\
  --pos_mode span --span_pre 2 --span_post 1 \\
  --epochs 3 --batch_queries 8 --lr 2e-5 \\
  --temperature_list 0.01,0.05,0.1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import lmk_train_data as D


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class LMKBiEncoder(nn.Module):
    """Encodes text with a HuggingFace transformer, inserts <LMK> as a special
    token, and uses the hidden state at the last <LMK> position as the
    sentence embedding

    """

    def __init__(
        self,
        model_name: str,
        max_length: int = 512,
        add_proj: bool = False,
        proj_dim: int = 256,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.encoder = AutoModel.from_pretrained(model_name)

        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<LMK>"]})
        self.encoder.resize_token_embeddings(len(self.tokenizer))

        #left-truncation: ensures the final <LMK> anchor survives trimming.
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "right"

        self.lmk_id = int(self.tokenizer.convert_tokens_to_ids("<LMK>"))
        self.max_length = int(max_length)

        hidden = int(self.encoder.config.hidden_size)
        self.proj: Optional[nn.Linear] = (
            nn.Linear(hidden, proj_dim, bias=False) if add_proj else None
        )

    def encode_texts(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """Encode a list of texts → L2-normalised (B, D) tensor."""
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)

        hs = self.encoder(**enc).last_hidden_state  # [B, T, H]
        input_ids = enc["input_ids"]

        vecs: List[torch.Tensor] = []
        for b in range(input_ids.size(0)):
            lmk_pos = (input_ids[b] == self.lmk_id).nonzero(as_tuple=False).flatten()
            idx = int(lmk_pos[-1].item()) if lmk_pos.numel() > 0 else 0
            vecs.append(hs[b, idx, :])

        V = torch.stack(vecs, dim=0)  # [B, H]
        if self.proj is not None:
            V = self.proj(V)
        return F.normalize(V, p=2, dim=-1)

    def save_checkpoint(self, directory: str) -> None:
        """Save encoder + tokenizer (+ optional projection head) to `directory`."""
        os.makedirs(directory, exist_ok=True)
        self.encoder.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        if self.proj is not None:
            torch.save(self.proj.state_dict(), os.path.join(directory, "proj_head.pt"))
            print(f"  [saved] proj_head.pt  ({self.proj.weight.shape})")


# ─────────────────────────────────────────────────────────────────────────────
# Freezing control
# ─────────────────────────────────────────────────────────────────────────────

def set_trainable_params(model: LMKBiEncoder, mode: str, last_n_layers: int = 2) -> None:
    """Set which parameters are trainable

    "lmk_only": only token embeddings
    "last_n":embeddings + last N transformer layers
    "all": full fine-tune
    """
    #start with everything frozen
    for p in model.encoder.parameters():
        p.requires_grad = False

    #Projection head always trainable.
    if model.proj is not None:
        for p in model.proj.parameters():
            p.requires_grad = True

    #Token embeddings always trainable
    for p in model.encoder.get_input_embeddings().parameters():
        p.requires_grad = True

    if mode == "lmk_only":
        return

    if mode == "all":
        for p in model.encoder.parameters():
            p.requires_grad = True
        return

    if mode == "last_n":
        enc = model.encoder
        #Locate the transformer layer list
        layers = None
        if hasattr(enc, "encoder") and hasattr(enc.encoder, "layer"):
            layers = enc.encoder.layer
        elif hasattr(enc, "transformer") and hasattr(enc.transformer, "layer"):
            layers = enc.transformer.layer
        elif hasattr(enc, "encoder") and hasattr(enc.encoder, "layers"):
            layers = enc.encoder.layers

        if layers is None:
            raise RuntimeError(
                "Cannot locate transformer layers for partial unfreeze.  "
                "Use --train_mode all or lmk_only instead."
            )
        n = len(layers)
        k = max(0, min(int(last_n_layers), n))
        for layer in layers[n - k :]:
            for p in layer.parameters():
                p.requires_grad = True
        return

    raise ValueError("train_mode must be one of: lmk_only, last_n, all")


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainExample:
    qid: str
    question: str
    gold_sentence_idx: int  # single int; multi-gold entries are skipped at load time


@dataclass
class PaperBundle:
    doc_id: str
    doc_path: str
    qa_module: str
    qa_var: str
    doc_index: D.DocIndex
    qa: List[TrainExample]
    mined: Optional[Dict[str, Any]]


@dataclass
class GlobalExample:
    bundle_idx: int
    qid: str
    question: str
    gold_sentence_idx: int


def load_paper_bundle(
    *,
    doc_id: str,
    doc_path: str,
    qa_module: str,
    qa_var: str,
    model_tokenizer: AutoTokenizer,
    model_name: str,
    max_length: int,
    window_size: int,
    mined_json: Optional[str],
) -> PaperBundle:
    text = D.load_text_file(doc_path)
    qa_raw = D.load_qa_raw(qa_module, qa_var)

    doc_index = D.build_doc_windows(
        doc_id=doc_id,
        raw_text=text,
        tokenizer=model_tokenizer,
        max_length=max_length,
        window_size=window_size,
        landmark_token="<LMK>",
        use_lmk=True,
    )

    qa: List[TrainExample] = []
    for item in qa_raw:
        qid = str(item.get("id", "")).strip()
        q = str(item.get("question", "")).strip()
        golds = D.gold_to_ints(item.get("gold_sentence_idx", None))
        #keeps multiple golds:
        #uses the first gold index for training so train/eval see the same questions
        if not qid or not q or not golds:
            continue
        qa.append(TrainExample(qid=qid, question=q, gold_sentence_idx=golds[0]))

    mined = D.load_mined_negatives(mined_json, expected_model=model_name) if mined_json else None
    return PaperBundle(
        doc_id=doc_id, doc_path=doc_path,
        qa_module=qa_module, qa_var=qa_var,
        doc_index=doc_index, qa=qa, mined=mined,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Candidate sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_candidates_for_query(
    bundle: PaperBundle,
    ex: TrainExample,
    *,
    pos_mode: str,
    span_pre: int,
    span_post: int,
    use_mined: bool,
    use_random_negs: bool,
    num_random_negs: int,
    num_pos_cap: int,
    num_mined_cap: int,
) -> Tuple[List[int], List[int]]:
    """Return (positive_anchor_indices, negative_anchor_indices) for one query"""
    n_sent = len(bundle.doc_index.sentences)

    pos = D.positive_anchor_set(
        ex.gold_sentence_idx, n_sent,
        mode=pos_mode, span_pre=span_pre, span_post=span_post,
    )
    if not pos:
        return [], []

    if len(pos) > num_pos_cap:
        pos = random.sample(pos, k=num_pos_cap)

    pos_set = set(pos)
    negs: List[int] = []

    if use_mined and bundle.mined is not None:
        rec = bundle.mined.get("items", {}).get(ex.qid)
        if rec is not None:
            mined_negs = [
                int(i) for i in rec.get("hard_neg_anchors", [])
                if int(i) not in pos_set
            ]
            negs.extend(mined_negs[:num_mined_cap])

    if use_random_negs:
        pool = [i for i in range(n_sent) if i not in pos_set]
        if pool:
            negs.extend(random.sample(pool, k=min(num_random_negs, len(pool))))

    #dedupe while preserving order
    negs = list(dict.fromkeys(negs))
    return pos, negs


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def multi_positive_infonce(
    sim: torch.Tensor,
    pos_mask: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Multi-positive InfoNCE loss

    Args:
        sim: [B, K] similarity matrix (dot products of L2-normalised vecs).
        pos_mask: [B, K] boolean mask, true where the candidate is positive.
        temperature: softmax temperature.

    Returns:
        Scalar mean loss over the batch.
    """
    sim_t = sim / temperature
    sim_t = sim_t - sim_t.max(dim=-1, keepdim=True).values  #numerical stability
    exp = torch.exp(sim_t)
    denom = exp.sum(dim=-1) + 1e-12
    numer = (exp * pos_mask.float()).sum(dim=-1) + 1e-12
    return (-torch.log(numer / denom)).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Training loop (single temperature)
# ─────────────────────────────────────────────────────────────────────────────

def run_training(
    model: LMKBiEncoder,
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
    """run the full training loop for a single temperature value"""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    steps_per_epoch = math.ceil(len(global_qa) / max(1, batch_queries) / max(1, grad_accum))
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    os.makedirs(out_dir, exist_ok=True)
    print(f"  temp={temperature:.4g}  steps={total_steps}  warmup={warmup_steps}")

    model.train()
    global_step = 0
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
                b = bundles[gx.bundle_idx]
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

                cand_idx = pos + negs
                cand_texts = [b.doc_index.window_texts[i] for i in cand_idx]
                pos_mask = [True] * len(pos) + [False] * len(negs)

                batch_q_texts.append(f"Question: {ex.question} <LMK>")
                batch_cand_texts.append(cand_texts)
                batch_pos_masks.append(pos_mask)

            if not batch_q_texts:
                continue

            q_vec = model.encode_texts(batch_q_texts, device=device)  #[B, D]

            if use_inbatch:
                #flatten all candidate texts across the batch.
                flat_cands: List[str] = []
                offsets: List[Tuple[int, int]] = []
                for cands in batch_cand_texts:
                    s = len(flat_cands)
                    flat_cands.extend(cands)
                    offsets.append((s, len(flat_cands)))

                d_vec = model.encode_texts(flat_cands, device=device)  #[K, D]
                sim = q_vec @ d_vec.T  #[B, K]

                pos_mask_t = torch.zeros(
                    (len(batch_q_texts), len(flat_cands)), dtype=torch.bool, device=device
                )
                for i, (s, e) in enumerate(offsets):
                    for j, is_pos in enumerate(batch_pos_masks[i]):
                        if is_pos:
                            pos_mask_t[i, s + j] = True

                loss = multi_positive_infonce(sim, pos_mask_t, temperature=temperature)
            else:
                losses: List[torch.Tensor] = []
                for i in range(len(batch_q_texts)):
                    d_vec = model.encode_texts(batch_cand_texts[i], device=device)
                    sim = q_vec[i : i + 1] @ d_vec.T  #[1, K_i]
                    pm = torch.tensor([batch_pos_masks[i]], dtype=torch.bool, device=device)
                    losses.append(multi_positive_infonce(sim, pm, temperature=temperature))
                loss = torch.stack(losses).mean()

            (loss / max(1, grad_accum)).backward()
            running_loss += float(loss.detach().cpu())

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
                    running_loss = 0.0
                    steps_since_log = 0 

        #flush any remaining gradients at end of epoch (handles non-divisible batches)
        if grad_accum > 1:
            remaining = (len(global_qa) // max(1, batch_queries)) % max(1, grad_accum)
            if remaining != 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

        #Save per-epoch checkpoint
        ckpt_dir = os.path.join(out_dir, f"epoch_{epoch + 1}")
        model.save_checkpoint(ckpt_dir)
        print(f"  [checkpoint] {ckpt_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ── Paper selection ──────────────
    ap.add_argument("--manifest", type=str, default="",
                    help="JSONL file with one paper config per line.")
    ap.add_argument("--paper_range", type=str, default="",
                    help="Paper index range, e.g. '1-24' or '1-6,9,12-14'.")
    ap.add_argument("--paper_exclude", type=str, default="",
                    help="Indices to exclude from --paper_range.")
    ap.add_argument("--doc_prefix", type=str, default="TEST_PAPER")
    ap.add_argument("--doc_ext", type=str, default=".mmd")
    ap.add_argument("--qa_prefix", type=str, default="QA_RAW")
    ap.add_argument("--qa_var", type=str, default="QA_RAW")
    ap.add_argument("--mined_mode", type=str, default="auto",
                    choices=["auto", "none", "require"])
    ap.add_argument("--mined_prefix", type=str, default="mined_TEST_PAPER")
    ap.add_argument("--mined_ext", type=str, default=".json")

    #single-paper fallback
    ap.add_argument("--doc_path", type=str, default="")
    ap.add_argument("--doc_id", type=str, default="DOC")
    ap.add_argument("--qa_module", type=str, default="")
    ap.add_argument("--mined_json", type=str, default="")

    #model / output
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="lmk_ft_out")

    # window / positives
    ap.add_argument("--window_size", type=int, default=10)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--pos_mode", type=str, default="span", choices=["span", "exact"])
    ap.add_argument("--span_pre", type=int, default=2)
    ap.add_argument("--span_post", type=int, default=1)

    # negatives
    ap.add_argument("--use_inbatch", action="store_true")
    ap.add_argument("--use_mined", action="store_true")
    ap.add_argument("--use_random_negs", action="store_true")
    ap.add_argument("--num_random_negs", type=int, default=16)
    ap.add_argument("--num_mined_cap", type=int, default=16)
    ap.add_argument("--num_pos_cap", type=int, default=4)

    #training hyperparameters 
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_queries", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument(
        "--temperature", type=float, default=0.05,
        help="InfoNCE temperature (used if --temperature_list is not given).",
    )
    ap.add_argument(
        "--temperature_list", type=str, default="",
        help="Comma-separated temperatures for a sweep, e.g. '0.01,0.05,0.1'. "
             "Each temperature gets its own output subdirectory.",
    )
    ap.add_argument("--seed", type=int, default=42)

    #Freezing mode 
    ap.add_argument("--train_mode", type=str, default="last_n",
                    choices=["lmk_only", "last_n", "all"])
    ap.add_argument("--train_last_n_layers", type=int, default=2)

    # optional projection head
    ap.add_argument("--add_proj", action="store_true")
    ap.add_argument("--proj_dim", type=int, default=256)

    args = ap.parse_args()

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
        idxs = [i for i in include if i not in exclude]
        if not idxs:
            raise ValueError("--paper_range produced no papers after exclude filter.")
        rows = D.build_rows_from_paper_indices(
            idxs,
            doc_prefix=args.doc_prefix, doc_ext=args.doc_ext,
            qa_prefix=args.qa_prefix, qa_var=args.qa_var,
            mined_prefix=args.mined_prefix, mined_ext=args.mined_ext,
            mined_mode=args.mined_mode,
        )
    else:
        if not args.doc_path or not args.qa_module:
            raise ValueError(
                "Provide --manifest, --paper_range, or (--doc_path and --qa_module)."
            )
        rows = [{
            "doc_id": args.doc_id,
            "doc_path": args.doc_path,
            "qa_module": args.qa_module,
            "qa_var": args.qa_var,
            "mined_json": args.mined_json or "",
        }]

    #Resolve temperature list 
    if args.temperature_list.strip():
        temperatures = [float(t.strip()) for t in args.temperature_list.split(",")]
    else:
        temperatures = [args.temperature]

    #Load bundles (once, shared across temperature sweep)
    #build a temporary model just to get the tokenizer for window construction.
    _tmp_model = LMKBiEncoder(
        model_name=args.model_name,
        max_length=args.max_length,
        add_proj=args.add_proj,
        proj_dim=args.proj_dim,
    )

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
            model_tokenizer=_tmp_model.tokenizer,
            model_name=args.model_name,
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
        raise RuntimeError("No QA examples loaded")

    print(f">>> Loaded {len(bundles)} papers, {len(global_qa)} total QA examples")
    for b in bundles:
        print(
            f"  {b.doc_id}: {len(b.qa)} QA, "
            f"{len(b.doc_index.sentences)} sents, "
            f"mined={'yes' if b.mined else 'no'}"
        )
    print(
        f">>> Negatives: inbatch={args.use_inbatch}  "
        f"mined={args.use_mined}  random={args.use_random_negs}"
    )
    print(f">>> train_mode={args.train_mode}  last_n={args.train_last_n_layers}")
    print(f">>> pos_mode={args.pos_mode}  span=({args.span_pre},{args.span_post})")
    print(f">>> Temperature sweep: {temperatures}")

    # Temperatre sweep
    for temp in temperatures:
        print(f"\n{'='*60}")
        print(f"  TEMPERATURE = {temp}")
        print(f"{'='*60}")

        #fresh model for each temperature
        model = LMKBiEncoder(
            model_name=args.model_name,
            max_length=args.max_length,
            add_proj=args.add_proj,
            proj_dim=args.proj_dim,
        ).to(device)
        set_trainable_params(model, mode=args.train_mode, last_n_layers=args.train_last_n_layers)

        #output dir
        if len(temperatures) > 1:
            temp_dir = os.path.join(args.out_dir, f"temp_{temp:.4g}")
        else:
            temp_dir = args.out_dir

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
