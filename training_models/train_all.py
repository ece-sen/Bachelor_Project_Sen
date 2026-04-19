import os
import sys
import random
from pathlib import Path
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation

from src.selector.schema_repr import load_schemas, load_queries, Preprocessor
from src.selector.lexical     import LexicalSelector
from src.selector.statistical import TFIDFSelector
from src.selector.semantical  import SemanticSelector
from mlp_fusion               import FusionMLP, FusionDataset, train_mlp
from hybrid                   import _minmax_normalize

# Reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


TRAIN_SPIDER_PATH   = "data/spider/train_spider.json"
TRAIN_OTHER_PATH = "data/spider/train_others.json"
DEV_PATH     = "data/spider/dev.json"
DATABASE_DIR = "data/spider/database"

# CKPT_FINETUNED = "models/gte-small-finetuned"
# CKPT_HARDNEG   = "models/gte-small-hardneg"
# CKPT_MLP       = "models/mlp_fusion.pt"

CKPT_FINETUNED = "models/gte-small-finetuned-v2"
CKPT_HARDNEG   = "models/gte-small-hardneg-v2"
CKPT_MLP       = "models/mlp_fusion-v2.pt"

BASE_MODEL     = "thenlper/gte-small"

# Hyperparameters
EPOCHS_SBERT   = 3
BATCH_SIZE     = 64
WARMUP_RATIO   = 0.1
NEG_HARD       = 7     # hard negatives per query for hard-neg fine-tune
NEG_MLP        = 5     # negatives per query for MLP dataset



# Shared helpers

def _warmup_steps(dataloader, epochs, ratio=WARMUP_RATIO):
    return int(len(dataloader) * epochs * ratio)


def _dev_evaluator(dev_qs, schemas):
    """
    Spearman similarity evaluator for epoch-end logging during SBERT training.
    Uses 500 dev queries (positive + one random negative each = 1000 pairs).
    """
    s1, s2, labels = [], [], []
    all_dbs = list(schemas.keys())

    for q in dev_qs[:500]:
        if q["db_id"] not in schemas:
            continue
        s1.append(q["question"]);  s2.append(schemas[q["db_id"]]);  labels.append(1.0)
        neg = random.choice([d for d in all_dbs if d != q["db_id"]])
        s1.append(q["question"]);  s2.append(schemas[neg]);          labels.append(0.0)

    return evaluation.EmbeddingSimilarityEvaluator(s1, s2, labels,
                                                   name="dev-similarity")


# Model 2 - GTE-small fine-tuned with random negatives

def build_random_neg_examples(train_qs, schemas):
    """
    One (query, correct_schema) pair per training query.
    MNRL constructs in-batch negatives automatically.
    """
    examples = []
    for q in train_qs:
        correct_db = q["db_id"]
        if correct_db not in schemas:
            continue
        examples.append(InputExample(
            texts=[q["question"], schemas[correct_db]]
        ))
    return examples


def train_finetuned(train_qs, dev_qs, schemas, output_dir):
    print("\n" + "=" * 60)
    print("Model 2 - GTE-small fine-tuned (random negatives)")
    print("=" * 60)

    examples   = build_random_neg_examples(train_qs, schemas)
    dataloader = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE)
    model      = SentenceTransformer(BASE_MODEL)
    loss       = losses.MultipleNegativesRankingLoss(model)
    evaluator  = _dev_evaluator(dev_qs, schemas)

    print(f"  Examples   : {len(examples)}")
    print(f"  Epochs     : {EPOCHS_SBERT}")
    print(f"  Output     : {output_dir}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.fit(
        train_objectives=[(dataloader, loss)],
        evaluator=evaluator,
        epochs=EPOCHS_SBERT,
        warmup_steps=_warmup_steps(dataloader, EPOCHS_SBERT),
        output_path=output_dir,
        save_best_model=True,
        show_progress_bar=True,
    )
    print(f"  Saved to {output_dir}")


# Model 3 - GTE-small fine-tuned with hard negatives

def mine_hard_negatives(train_qs, schemas, bm25, tfidf, neg_per_query):
    all_dbs  = list(schemas.keys())
    examples = []

    print("  Mining hard negatives with BM25 + TF-IDF...")
    for i, q in enumerate(train_qs):
        if i % 1000 == 0:
            print(f"    {i}/{len(train_qs)}")

        correct_db = q["db_id"]
        question   = q["question"]

        if correct_db not in schemas:
            continue

        bm25_scores  = _minmax_normalize(bm25.score(question))
        tfidf_scores = _minmax_normalize(tfidf.score(question))
        fused = {
            db: (bm25_scores.get(db, 0.0) + tfidf_scores.get(db, 0.0)) / 2
            for db in all_dbs if db != correct_db
        }
        hard_negs = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        hard_negs = [db for db, _ in hard_negs[:neg_per_query]]

        # triplet: (query, positive, hard_negative)
        # one InputExample per hard negative
        for hard_neg_db in hard_negs:
            examples.append(InputExample(
                texts=[question, schemas[correct_db], schemas[hard_neg_db]]
            ))

    return examples

def train_hardneg(train_qs, dev_qs, schemas, bm25, tfidf, output_dir):
    print("\n" + "=" * 60)
    print("Model 3 - GTE-small fine-tuned (hard negatives)")
    print("=" * 60)

    examples   = mine_hard_negatives(train_qs, schemas, bm25, tfidf, NEG_HARD)

    HARD_NEG_BATCH = 16 # reduced bc of memory issues in kaggle

    dataloader = DataLoader(examples, shuffle=True, batch_size=HARD_NEG_BATCH)
    model      = SentenceTransformer(BASE_MODEL)
    # Enable gradient checkpointing to save memory
    model[0].auto_model.gradient_checkpointing_enable()
    loss       = losses.MultipleNegativesRankingLoss (model)
    evaluator  = _dev_evaluator(dev_qs, schemas)

    print(f"  Examples   : {len(examples)}")
    print(f"  Epochs     : {EPOCHS_SBERT}")
    print(f"  Output     : {output_dir}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.fit(
        train_objectives=[(dataloader, loss)],
        evaluator=evaluator,
        epochs=EPOCHS_SBERT,
        warmup_steps=_warmup_steps(dataloader, EPOCHS_SBERT),
        output_path=output_dir,
        save_best_model=True,
        show_progress_bar=True,
    )
    print(f"  Saved to {output_dir}")


# Main

if __name__ == "__main__":

    print("Loading schemas and queries...")
    p                    = Preprocessor(remove_generic=True, lemmatize=True)
    schemas_preprocessed = load_schemas(DATABASE_DIR, preprocessor=p)
    raw_schemas          = load_schemas(DATABASE_DIR)
    train_spider_qs      = load_queries(TRAIN_SPIDER_PATH)
    train_other_qs       = load_queries(TRAIN_OTHER_PATH)
    train_qs             = train_spider_qs + train_other_qs
    dev_qs               = load_queries(DEV_PATH)

    print(f"  Train queries : {len(train_qs)}")
    print(f"  Dev queries   : {len(dev_qs)}")
    print(f"  Schemas       : {len(raw_schemas)}")

    print("\nInitializing base selectors (BM25, TF-IDF, GTE-small)...")
    bm25     = LexicalSelector(schemas_preprocessed, preprocessor=p, variant="okapi")
    tfidf    = TFIDFSelector(schemas_preprocessed, preprocessor=p, ngram_range=(1, 2))
    semantic = SemanticSelector(raw_schemas, model_name=BASE_MODEL)

    print("\n" + "=" * 60)
    print("Model 1 - GTE-small base (no training, used as baseline)")
    print("=" * 60)
    print("  Nothing to train - loaded at eval time from HuggingFace.")

    # Model 2: positives only + CachedMNRL
    train_finetuned(train_qs, dev_qs, raw_schemas, CKPT_FINETUNED)

    # Free memory from Model 2 before training Model 3
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

    # Model 3: hard negative triplets + CachedMNRL
    train_hardneg(train_qs, dev_qs, raw_schemas, bm25, tfidf, CKPT_HARDNEG)

    # Model 4: MLP - pass schemas_preprocessed, not raw_schemas
    print("\n" + "=" * 60)
    print("Model 4 - MLP fusion (BM25 + TF-IDF + GTE-small)")
    print("=" * 60)
    train_mlp(bm25, tfidf, semantic, schemas_preprocessed, train_qs,
              model_path=CKPT_MLP, neg_per_query=NEG_MLP)

    print("\n" + "=" * 60)
    print("All models trained. Run evaluation script to test on test set.")
    print("=" * 60)