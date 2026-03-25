def evaluate(selector, queries: list) -> dict:
    """
    Runs selector on all queries and returns accuracy/MRR metrics.
    """
    top1_correct = 0
    top3_correct = 0
    reciprocal_ranks = []

    for q in queries:
        question = q["question"]
        correct_db = q["db_id"]
        ranked = selector.rank(question, top_k=3)
        ranked_ids = [db for db, _ in ranked]

        if ranked_ids[0] == correct_db:
            top1_correct += 1
        if correct_db in ranked_ids:
            top3_correct += 1

        rank = next((i + 1 for i, db in enumerate(ranked_ids) if db == correct_db), None)
        reciprocal_ranks.append(1 / rank if rank else 0.0)

    total = len(queries)
    return {
        "top1": top1_correct / total,
        "top3": top3_correct / total,
        "mrr":  sum(reciprocal_ranks) / total,
    }