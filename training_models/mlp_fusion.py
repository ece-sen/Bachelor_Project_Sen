"""
RQ2 — Learned Score Fusion via MLP

Architecture
------------
For every (query, db_id) pair the three selectors each produce one
scalar score.  Those three numbers are stacked into a feature vector
and passed through a small MLP that outputs a single relevance score:

    [bm25_norm, tfidf_norm, semantic_norm]  →  MLP  →  relevance ∈ (0, 1)

The MLP is trained with BCELoss (binary cross-entropy):
    label = 1.0  if db_id == correct_db   (positive)
    label = 0.0  otherwise                (negative)

At inference the MLP score replaces the manual weights from hybrid.py,
so this is a direct learned alternative to the grid-searched hybrid.

Why this design?
- Three input features only → tiny model, no overfitting risk
- Scores are already normalized → stable gradients, no need for BatchNorm
- BCELoss + sigmoid output → directly interpretable as P(correct DB)
- Trained on Spider train set, evaluated on dev set → clean split

File layout
-----------
    mlp_fusion.py           ← this file (model + training + selector)
    models/mlp_fusion.pt    ← saved MLP weights after training
"""

import os
import sys
import random
import json
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.selector.schema_repr import load_schemas, load_queries, Preprocessor
from src.selector.lexical     import LexicalSelector
from src.selector.statistical import TFIDFSelector
from src.selector.semantical  import SemanticSelector
from src.evaluation.metrics   import evaluate
from hybrid import _minmax_normalize   # reuse, never duplicate


# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = "models/mlp_fusion.pt"
TRAIN_PATH   = "data/spider/train_spider.json"
DEV_PATH     = "data/spider/dev.json"
DATABASE_DIR = "data/spider/database"

NEG_PER_QUERY = 5      # negatives sampled per query during training
                        # more than finetune.py because MLP is fast to train
EPOCHS        = 20
BATCH_SIZE    = 64
LR            = 1e-3


# ── MLP definition ─────────────────────────────────────────────────────────────

class FusionMLP(nn.Module):
    """
    3 → 32 → 16 → 1  feed-forward network with ReLU activations.

    Input  : [bm25_norm, tfidf_norm, semantic_norm]  (all in [0, 1])
    Output : relevance score passed through sigmoid → scalar ∈ (0, 1)

    Architecture kept deliberately small:
    - Only 3 input features → deep network would overfit immediately
    - Hidden layers add non-linearity so the MLP can learn e.g.
      "high semantic AND low BM25 is still a strong signal"
      which a linear weighted sum (hybrid.py) cannot express
    - Dropout(0.3) on the first hidden layer regularises training
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Dataset ────────────────────────────────────────────────────────────────────

class FusionDataset(Dataset):
    """
    Pre-computes all three normalized scores for every
    (query, db_id) training pair and stores them as tensors.

    Pre-computing is much faster than scoring on-the-fly inside
    the training loop because the BM25/TF-IDF/semantic selectors
    are called once per query, not once per batch.

    Each item: (feature_tensor [3], label_tensor [1])
    """

    def __init__(
        self,
        queries:   list,
        schemas:   dict,
        bm25:      LexicalSelector,
        tfidf:     TFIDFSelector,
        semantic:  SemanticSelector,
        neg_per_query: int = NEG_PER_QUERY,
    ):
        self.features: list[torch.Tensor] = []
        self.labels:   list[float]        = []

        all_db_ids = list(schemas.keys())

        print("Pre-computing scores for training set...")
        for i, q in enumerate(queries):
            if i % 500 == 0:
                print(f"  {i}/{len(queries)}")

            question   = q["question"]
            correct_db = q["db_id"]

            if correct_db not in schemas:
                continue

            # score ALL databases once per query
            bm25_scores     = _minmax_normalize(bm25.score(question))
            tfidf_scores    = _minmax_normalize(tfidf.score(question))
            semantic_scores = _minmax_normalize(semantic.score(question))

            # positive pair
            self._add(bm25_scores, tfidf_scores, semantic_scores,
                      correct_db, label=1.0)

            # negative pairs — random sample
            wrong_dbs = [db for db in all_db_ids if db != correct_db]
            for db_id in random.sample(wrong_dbs,
                                       min(neg_per_query, len(wrong_dbs))):
                self._add(bm25_scores, tfidf_scores, semantic_scores,
                          db_id, label=0.0)

        print(f"  Done. {len(self.features)} total examples.\n")

    def _add(self, bm25_scores, tfidf_scores, semantic_scores,
             db_id: str, label: float):
        feat = torch.tensor([
            bm25_scores.get(db_id, 0.0),
            tfidf_scores.get(db_id, 0.0),
            semantic_scores.get(db_id, 0.0),
        ], dtype=torch.float32)
        self.features.append(feat)
        self.labels.append(label)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], torch.tensor(self.labels[idx],
                                                dtype=torch.float32)


# ── Training loop ──────────────────────────────────────────────────────────────

def train_mlp(
    bm25:     LexicalSelector,
    tfidf:    TFIDFSelector,
    semantic: SemanticSelector,
    schemas:  dict,
    queries:  list,
    epochs:      int   = EPOCHS,
    batch_size:  int   = BATCH_SIZE,
    lr:          float = LR,
    model_path:  str   = MODEL_PATH,
) -> FusionMLP:

    dataset    = FusionDataset(queries, schemas, bm25, tfidf, semantic)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model     = FusionMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    print(f"Training MLP for {epochs} epochs...")
    print(f"{'Epoch':>6} {'Loss':>10}")
    print("-" * 18)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for features, labels in dataloader:
            optimizer.zero_grad()
            preds = model(features)
            loss  = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        avg_loss = total_loss / len(dataset)
        print(f"{epoch:>6} {avg_loss:>10.4f}")

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\nMLP saved to {model_path}")

    return model


# ── Inference selector ─────────────────────────────────────────────────────────

class MLPFusionSelector:
    """
    Drop-in selector that uses the trained MLP to score every database.

    Implements the same .score() / .rank() interface as all other
    selectors so it works directly with evaluate() from metrics.py
    and can be slotted into hybrid.py if needed.

    Usage:
        selector = MLPFusionSelector(bm25, tfidf, semantic, model_path)
        selector.rank("How many singers are there?", top_k=3)
    """

    def __init__(
        self,
        bm25:      LexicalSelector,
        tfidf:     TFIDFSelector,
        semantic:  SemanticSelector,
        model_path: str = MODEL_PATH,
    ):
        self.bm25     = bm25
        self.tfidf    = tfidf
        self.semantic = semantic
        self.db_ids   = list(bm25.db_ids)   # all three share the same db_ids

        self.mlp = FusionMLP()
        self.mlp.load_state_dict(torch.load(model_path, weights_only=True))
        self.mlp.eval()

    def score(self, query: str) -> dict:
        bm25_scores     = _minmax_normalize(self.bm25.score(query))
        tfidf_scores    = _minmax_normalize(self.tfidf.score(query))
        semantic_scores = _minmax_normalize(self.semantic.score(query))

        results = {}
        with torch.no_grad():
            for db_id in self.db_ids:
                feat = torch.tensor([
                    bm25_scores.get(db_id, 0.0),
                    tfidf_scores.get(db_id, 0.0),
                    semantic_scores.get(db_id, 0.0),
                ], dtype=torch.float32)
                results[db_id] = float(self.mlp(feat.unsqueeze(0)))

        return results

    def rank(self, query: str, top_k: int = 3) -> list:
        scores = self.score(query)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading schemas and queries...")
    p                    = Preprocessor(remove_generic=True, lemmatize=True)
    schemas_preprocessed = load_schemas(DATABASE_DIR, preprocessor=p)
    raw_schemas          = load_schemas(DATABASE_DIR)
    train_qs             = load_queries(TRAIN_PATH)
    dev_qs               = load_queries(DEV_PATH)

    # ── Build base selectors ───────────────────────────────────────────────
    print("Initializing base selectors...")
    bm25     = LexicalSelector(schemas_preprocessed, preprocessor=p, variant="okapi")
    tfidf    = TFIDFSelector(schemas_preprocessed, preprocessor=p,
                             ngram_range=(1, 2))
    semantic = SemanticSelector(raw_schemas, model_name="thenlper/gte-small")

    # ── Train MLP ──────────────────────────────────────────────────────────
    train_mlp(bm25, tfidf, semantic, raw_schemas, train_qs)

    # ── Evaluate on dev ────────────────────────────────────────────────────
    print("\nEvaluating MLP fusion on dev set...")
    selector = MLPFusionSelector(bm25, tfidf, semantic, MODEL_PATH)
    r        = evaluate(selector, dev_qs)
    print(f"Top-1: {r['top1']:.3f}  Top-3: {r['top3']:.3f}  MRR: {r['mrr']:.3f}")