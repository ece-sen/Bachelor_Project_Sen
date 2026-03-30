def compute_mrr(ranked_ids: list, correct_db: str) -> float:
    rank = next(
        (i + 1 for i, db in enumerate(ranked_ids) if db == correct_db),
        None
    )
    return 1 / rank if rank else 0.0

def evaluate(selector, queries: list) -> dict:
    """
    Runs selector on all queries and returns accuracy/MRR metrics.
    """
    top1_correct = 0
    top3_correct = 0
    reciprocal_ranks_3 = []
    reciprocal_ranks_10 = []

    for q in queries:
        question = q["question"]
        correct_db = q["db_id"]
        ranked_3 = selector.rank(question, top_k=3)
        ranked_ids = [db for db, _ in ranked_3]

        if ranked_ids[0] == correct_db:
            top1_correct += 1
        if correct_db in ranked_ids:
            top3_correct += 1

        reciprocal_ranks_3.append(compute_mrr(ranked_ids, correct_db))

        ranked_10  = selector.rank(question, top_k=10)
        mrr_ids_10 = [db for db, _ in ranked_10]
        reciprocal_ranks_10.append(compute_mrr(mrr_ids_10, correct_db))


    total = len(queries)
    return {
        "top1": top1_correct / total,
        "top3": top3_correct / total,
        "mrr@3":  sum(reciprocal_ranks_3) / total,
        "mrr@10": sum(reciprocal_ranks_10) / total,
    }