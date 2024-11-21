from typing import List
from fuzzyops.fuzzy_numbers import FuzzyNumber


def fuzzy_sum_solver(criteria_weights: List[FuzzyNumber],
                     alternatives_scores: List[List[FuzzyNumber]]) -> List[FuzzyNumber]:
    """
    Вычисляет взвешенную сумму оценок для альтернатив на основе заданных весов критериев.

    Args:
        criteria_weights (List[FuzzyNumber]): Список нечетких чисел, представляющий веса критериев.
        alternatives_scores (List[List[FuzzyNumber]]): Двумерный список нечетких чисел, представляющий оценки для
        каждой альтернативы по каждому критерию.

    Returns:
        List[FuzzyNumber]: Список нечетких чисел, представляющий итоговые взвешенные суммы для каждой альтернативы.

    Raises:
        ValueError: Если количество оценок для одной из альтернатив не соответствует количеству критериев.
    """

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
