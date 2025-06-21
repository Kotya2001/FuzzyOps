from typing import List
from fuzzyops.fuzzy_numbers import FuzzyNumber


def fuzzy_sum_solver(criteria_weights: List[FuzzyNumber],
                     alternatives_scores: List[List[FuzzyNumber]]) -> List[FuzzyNumber]:
    """
    Calculates a weighted sum of estimates for alternatives based on the given weights of criteria

    Args:
        criteria_weights (List[FuzzyNumber]): A list of fuzzy numbers representing the weights of the criteria
        alternatives_scores (List[List[FuzzyNumber]]): A two-dimensional list of fuzzy numbers representing the scores for
        each alternative according to each criterion

    Returns:
        List[FuzzyNumber]: A list of fuzzy numbers representing the final weighted sums for each alternative

    Raises:
        ValueError: If the number of ratings for one of the alternatives does not match the number of criteria
    """

    # Проверим, что число критериев совпадает с числом оценок по каждому критерию
    num_criteria = len(criteria_weights)
    for scores in alternatives_scores:
        if len(scores) != num_criteria:
            raise ValueError("The number of ratings does not match the number of criteria")

    # Вычисляем взвешенную сумму для каждой альтернативы
    total_scores = []
    for _, scores in enumerate(alternatives_scores):
        total = sum(w * s for w, s in zip(criteria_weights, scores))
        total_scores.append(total)

    return total_scores
