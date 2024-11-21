from typing import List
from fuzzyops.fuzzy_numbers import FuzzyNumber


def fuzzy_hierarchy_solver(criteria_weights: List[List[FuzzyNumber]],
                           alternative_comparisons: List[List[List[FuzzyNumber]]]) -> List[FuzzyNumber]:
    """
    Решатель нечеткой иерархии, использующий метод анализа иерархий для определения приоритетов альтернатив.

    Args:
        criteria_weights (List[List[FuzzyNumber]]): Двумерный список нечетких чисел, представляющий веса критериев.
        alternative_comparisons (List[List[List[FuzzyNumber]]]): Трехмерный список нечетких чисел, представляющий
        сравнительные оценки альтернатив для каждого критерия.

    Returns:
        List[FuzzyNumber]: Список нечетких чисел, представляющий глобальные приоритеты альтернатив.
    """

    def normalize_matrix(matrix: List[List[FuzzyNumber]]) -> List[float]:
        """
        Нормализует переданную матрицу нечетких чисел и вычисляет приоритеты.

        Args:
            matrix (List[List[FuzzyNumber]]): Двумерный список нечетких чисел, представляющий матрицу сравнений.

        Returns:
            List[float]: Список нормализованных приоритетов для строк матрицы.
        """

        n = len(matrix)
        col_sums = [sum([matrix[i][j] for i in range(n)]) for j in range(n)]
        normalized_matrix = [[matrix[i][j] / float(col_sums[j]) for j in range(n)] for i in range(n)]
        priorities = [sum(row) / n for row in normalized_matrix]
        return priorities

    criteria_priorities = normalize_matrix(criteria_weights)

    alternatives_priorities = []
    for comparison in alternative_comparisons:
        priorities = normalize_matrix(comparison)
        alternatives_priorities.append(priorities)

    num_alternatives = len(alternative_comparisons[0])
    global_priorities = [0] * num_alternatives
    for i in range(num_alternatives):
        for j in range(len(criteria_priorities)):
            global_priorities[i] += alternatives_priorities[j][i] * criteria_priorities[j]

    return global_priorities
