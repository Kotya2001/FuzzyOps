from typing import List, Tuple
from fuzzyops.fuzzy_numbers import FuzzyNumber


def fuzzy_pairwise_solver(alternatives: List[str], criteria: List[str],
                          pairwise_matrices: List[List[List[FuzzyNumber]]]) -> List[Tuple[str, FuzzyNumber]]:
    """
    Решатель сравнений парных нечетких альтернатив, использующий метод анализа для оценки и ранжирования.

    Args:
        alternatives (List[str]): Список строк, представляющих названия альтернатив.
        criteria (List[str]): Список строк, представляющих названия критериев.
        pairwise_matrices (List[List[List[FuzzyNumber]]]): Список матриц парных сравнений для каждой альтернативы
        по каждому критерию, содержащих нечеткие числа.

    Returns:
        List[Tuple[str, FuzzyNumber]]: Список кортежей, каждый из которых содержит название альтернативы и
        соответствующий ей итоговую нечеткую оценку, отсортированные по убыванию.

    Raises:
        ValueError: Если количество матриц парных сравнений не совпадает с количеством критериев.
    """

    num_alternatives = len(alternatives)
    num_criteria = len(criteria)

    if len(pairwise_matrices) != num_criteria:
        raise ValueError("Число матриц парных сравнений должно совпадать с числом критериев.")

    def normalize_matrix(matrix: List[List[FuzzyNumber]]) -> List[List[float]]:
        """
        Нормализует переданную матрицу нечетких чисел.

        Args:
            matrix (List[List[FuzzyNumber]]): Двумерный список нечетких чисел, представляющий матрицу парных сравнений.

        Returns:
            List[List[float]]: Нормализованная матрица, где каждое значение является
            пропорцией по сравнению с суммой соответствующего столбца.
        """

        col_sum = [sum(col) for col in zip(*matrix)]
        normalized_matrix = [[matrix[i][j] / float(col_sum[j]) for j in range(len(matrix[i]))] for i in
                             range(len(matrix))]
        return normalized_matrix

    def get_weights(normalized_matrix: List[List[FuzzyNumber]]) -> List[FuzzyNumber]:
        """
        Вычисляет веса для каждой строки в нормализованной матрице.

        Args:
            normalized_matrix (List[List[FuzzyNumber]]): Нормализованная матрица парных сравнений.

        Returns:
            List[FuzzyNumber]: Список весов, представляющий среднее значение по каждой строке.
        """

        row_averages = [sum(row) / len(row) for row in normalized_matrix]
        return row_averages

    alternative_scores = [0] * num_alternatives

    for k in range(num_criteria):
        matrix = pairwise_matrices[k]

        normalized_matrix = normalize_matrix(matrix)

        weights = get_weights(normalized_matrix)

        for i in range(num_alternatives):
            alternative_scores[i] += weights[i]

    total_score_sum = sum(alternative_scores)
    final_scores = [score / float(total_score_sum) for score in alternative_scores]

    ranked_alternatives = sorted(zip(alternatives, final_scores), key=lambda x: float(x[1]), reverse=True)

    return ranked_alternatives
