

def fuzzy_sum_solver(criteria_weights, alternatives_scores):
    # Проверим, что число критериев совпадает с числом оценок по каждому критерию
    num_criteria = len(criteria_weights)
    for scores in alternatives_scores:
        if len(scores) != num_criteria:
            raise ValueError("Количество оценок не соответствует количеству критериев.")

    # Вычисляем взвешенную сумму для каждой альтернативы
    total_scores = []
    for scores in alternatives_scores:
        total = sum(w * s for w, s in zip(criteria_weights, scores))
        total_scores.append(total)

    return total_scores