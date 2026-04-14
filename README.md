# LMK Embedding Pipeline

Dense retrieval pipeline for ChATLAS — a physics paper Q&A assistant built on the ATLAS publication corpus. This repo contains every stage of the pipeline: scraping, training, hard-negative mining, evaluation, and inference, for a contrastively fine-tuned bi-encoder. Three pooling strategies are compared: LMK (landmark token), CLS, and mean pooling.

# Motivation
Physics literature presents retrieval challenges that general-purpose models are not equipped to handle: dense mathematical notation, heavy use of domain-specific acronyms (e.g. ATLAS, LHC, BSM), cross-paper dependencies where answers span multiple publications, and sentence structures that embed quantitative claims inside complex experimental context. Standard bi-encoders trained on general corpora systematically fail on these queries, motivating a domain-adapted approach trained specifically on ATLAS publication data.

---

## What is LMK pooling?

Standard bi-encoders compress a passage into a single vector using CLS (first token) or mean pooling. LMK pooling instead injects a special `<LMK>` token into the input at the end of each sentence boundary, and trains the model to aggregate passage information into that token's hidden state. This gives the model an explicit, learnable aggregation site:

- **CLS** — position-biased toward the start of the sequence.
- **Mean** — diluted by uninformative tokens.
- **LMK** — dedicated token placed at the anchor sentence, trained to summarise the surrounding context window.

During retrieval each document sentence becomes an *anchor*: its context window of up to `window_size` preceding sentences is encoded, and the `<LMK>` hidden state at the anchor position is used as the passage embedding.

---

## Key results

| Model | Dense Acc@1 | Acc@5 | Acc@10 | CE Acc@1 |
|---|---|---|---|---|
| LMK_FT16 (w=10 eval) | **88.7%** | 98.7% | 99.8% | 79.7% |
| LMK_FT16 (w=3 eval)  | 75.5%     | 92.4% | 96.9% | 79.7% |
| CLS_FT16 (w=3 eval)  | 58.3%     | 82.4% | 89.3% | 79.9% |
| MEAN_FT16 (w=3 eval) | 64.6%     | 88.4% | 92.6% | 79.9% |

All models fine-tuned for 16 epochs with in-batch negatives + hard negatives. Base model: `sentence-transformers/all-mpnet-base-v2`.

---

## Pipeline overview

```
CDS (CERN Document Server)
        │
        ▼
Scraping/cds_scrape.py          ← download PDFs, run Nougat OCR → .mmd files
        │
        ▼
Scraping/split_sentences.py     ← inspect sentence splits, assign gold_sentence_idx
        │
        ▼
Training/train_lmk_retriever.py         ← initial training run (warm-up)
Training/train_pooling_retriever.py     ← CLS / mean baseline training
        │
        ▼
Training/mine_hard_negatives.py  ← mine hard negatives from warm-up checkpoint
        │
        ▼
Training/train_lmk_retriever.py  ← final training with mined hard negatives
        │
        ▼
Eval/eval_comprehensive.py       ← Acc@K, MRR, NDCG — dense + CE reranked
Eval/eval_bootstrap.py           ← bootstrap confidence intervals
Eval/plot_results_v3.py          ← ablation plots
Eval/plot_tsne.py                ← t-SNE embedding visualisations
Eval/plot_embedding_shift.py     ← per-token embedding shift analysis
        │
        ▼
Embedding/LMK_Embed10_DENSE_ONLY.py  ← interactive QA over a document
```

---

## Repo structure

```
Scraping/
    cds_scrape.py               # download ATLAS papers from CDS, OCR with Nougat
    split_sentences.py          # print numbered sentences for gold annotation

Training/
    lmk_train_data.py           # shared utilities: sentence splitting, window
                                #   construction, DocIndex, hard-neg I/O
    train_lmk_retriever.py      # contrastive fine-tuning with LMK pooling
    train_pooling_retriever.py  # contrastive fine-tuning with CLS or mean pooling
    mine_hard_negatives.py      # FAISS-free hard negative mining via cosine search

Eval/
    eval_comprehensive.py       # main eval script: Acc@K, MRR, NDCG (dense + CE)
    eval_bootstrap.py           # bootstrap CI across papers
    plot_results_v3.py          # ablation plots (layers, negatives, temperature, seeds)
    plot_tsne.py                # t-SNE of base vs fine-tuned embeddings
    plot_embedding_shift.py     # per-token embedding shift analysis

Embedding/
    LMK_Embed10_DENSE_ONLY.py   # universal Encoder class (lmk/cls/mean) + interactive QA
```

---

## Installation

```bash
# Core dependencies
pip install torch transformers numpy

# For scraping (Nougat OCR)
module load CMake/3.21.1-GCCcore-11.2.0  # HPC only
uv pip install --only-binary=:all: "pyarrow==14.0.2"
uv pip install \
  "sentencepiece==0.1.99" \
  "nougat-ocr==0.1.17" \
  "pypdfium2==4.30.0" \
  PyPDF2 requests beautifulsoup4 \
  "transformers==4.35.0" \
  "tokenizers==0.14.1" \
  "huggingface-hub==0.17.3" \
  "albumentations==1.3.1" \
  "opencv-python-headless<5.0.0.0"
uv pip install --force-reinstall "numpy<2"
```

---

## Data format

Each paper requires two files in the working directory:

| File | Description |
|---|---|
| `TEST_PAPER{N}.mmd` | Nougat OCR output (markdown) for paper N |
| `QA_RAW{N}.py` | Python module exporting a `QA_RAW` list of dicts |

Each entry in `QA_RAW` must have:
```python
{
    "id": "q1",
    "question": "What is the signal region definition?",
    "gold_sentence_idx": 42,   # int or list[int]
}
```

Hard-negative JSON files (produced by `mine_hard_negatives.py`) are named `mined_TEST_PAPER{N}.json` by default and are picked up automatically.

---

## Usage

### 1. Scrape and OCR papers

```bash
python Scraping/cds_scrape.py \
    --n_docs 50 \
    --start_index 1 \
    --base_dir data/papers/
```

### 2. Inspect sentence splits (for gold annotation)

```bash
# Print sentences 0–99 from paper 1
python Scraping/split_sentences.py --paper_id 1 --start 0 --end 100

# Explicit input/output
python Scraping/split_sentences.py \
    --input TEST_PAPER1.mmd \
    --output TEST_PAPER1_sentences.txt
```

### 3. Initial training (LMK)

```bash
python Training/train_lmk_retriever.py \
    --paper_range 1-24 \
    --model_name sentence-transformers/all-mpnet-base-v2 \
    --out_dir ft_runs/warmup \
    --use_inbatch \
    --train_mode last_n --train_last_n_layers 2 \
    --pos_mode span --span_pre 2 --span_post 1 \
    --epochs 8 --batch_queries 8 --lr 2e-5 --temp 0.07
```

Training modes (`--train_mode`):
- `lmk_only` — only the new `<LMK>` token embedding is trained
- `last_n` — last N transformer layers + embeddings (default, recommended)
- `all` — full fine-tune

### 4. Mine hard negatives

```bash
python Training/mine_hard_negatives.py \
    --paper_range 1-24 \
    --model_name ft_runs/warmup/epoch_8 \
    --window_size 10 \
    --num_hard_negs 20
```

Outputs `mined_TEST_PAPER{N}.json` for each paper.

### 5. Final training with hard negatives

```bash
python Training/train_lmk_retriever.py \
    --paper_range 1-24 \
    --model_name sentence-transformers/all-mpnet-base-v2 \
    --out_dir ft_runs/final \
    --use_inbatch --use_mined \
    --train_mode last_n --train_last_n_layers 2 \
    --pos_mode span --span_pre 2 --span_post 1 \
    --epochs 16 --batch_queries 8 --lr 2e-5 --temp 0.07
```

Temperature sweep (saves a separate subdirectory per temperature):
```bash
python Training/train_lmk_retriever.py \
    --paper_range 1-24 \
    --model_name sentence-transformers/all-mpnet-base-v2 \
    --out_dir ft_runs/sweep \
    --use_inbatch --use_mined \
    --epochs 8 \
    --temperature_list 0.01,0.05,0.07,0.1
```

### 6. Train CLS / mean baseline

```bash
python Training/train_pooling_retriever.py \
    --paper_range 1-24 \
    --model_name sentence-transformers/all-mpnet-base-v2 \
    --pooling cls \
    --out_dir ft_runs/cls_baseline \
    --use_inbatch --use_mined \
    --epochs 16 --lr 2e-5
```

### 7. Evaluate

```bash
python Eval/eval_comprehensive.py \
    --paper_range 1-24 \
    --model LMK_FT:ft_runs/final/epoch_16:lmk \
    --model CLS_FT:ft_runs/cls_baseline/epoch_16:cls \
    --model MEAN_BASE:sentence-transformers/all-mpnet-base-v2:mean \
    --ce_model cross-encoder/ms-marco-MiniLM-L-6-v2 \
    --window_size 10 \
    --k_max 10 \
    --out_csv results/eval.csv \
    --out_json results/eval.json
```

Model spec format: `name:model_path_or_hf_id:pooling_mode`
Pooling modes: `lmk`, `cls`, `mean`

### 8. Interactive QA (single document)

```bash
python Embedding/LMK_Embed10_DENSE_ONLY.py \
    --doc_path TEST_PAPER1.mmd \
    --model_name ft_runs/final/epoch_16 \
    --pooling_mode lmk \
    --window_size 10
```

Type questions at the `Question>` prompt. Add `--no_ce` to skip cross-encoder reranking.

---

## Notes

- **GPU required for training.** Scripts are set up for SLURM (`export CUDA_VISIBLE_DEVICES=0`).
- **LMK pooling is sensitive to window size.** Retraining with a smaller window (e.g. `window_size=3`) hurts LMK significantly more than CLS or mean, because the anchor `<LMK>` has less context to aggregate. Use `window_size=10` for best LMK performance.
- **Two-stage training is recommended.** Run a warm-up first, mine hard negatives from that checkpoint, then retrain with `--use_mined`. The mined negatives JSON stores the model name used to mine them; a warning is printed if the model changes.
- **Left-truncation.** The tokeniser is configured with `truncation_side="left"` so that the final `<LMK>` anchor always survives truncation of long windows.
- **Sentence index alignment.** `split_into_sentences` uses a fixed regex (`(?<=[.!?])\s+`). Do not modify it without re-annotating all `QA_RAW` files, as `gold_sentence_idx` values must stay aligned.

---

## Context

This pipeline was developed as part of a dissertation project on dense retrieval for high-energy physics literature (ChATLAS). The ATLAS corpus was scraped from the CERN Document Server (CDS), each split into overlapping sentence windows used as retrieval passages.
