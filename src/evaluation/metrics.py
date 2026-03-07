def evaluate(selector, queries: list) -> tuple:
    """
    Runs selector on all queries and returns (top1_accuracy, top3_accuracy).
    Reusable for any selector that has a .rank() method.
    Works with LexicalSelector, TFIDFSelector, SemanticSelector, HybridSelector.
    """
    top1_correct = 0
    top3_correct = 0

    for q in queries:
        question = q["question"]
        correct_db = q["db_id"]
        ranked = selector.rank(question, top_k=3)

        if ranked[0][0] == correct_db:
            top1_correct += 1
        if correct_db in [db for db, _ in ranked]:
            top3_correct += 1

    total = len(queries)
    return top1_correct / total, top3_correct / total