from typing import List, Tuple
from fuzzyops.fuzzy_numbers import FuzzyNumber


def fuzzy_pairwise_solver(alternatives: List[str], criteria: List[str],
                          pairwise_matrices: List[List[List[FuzzyNumber]]]) -> List[Tuple[str, FuzzyNumber]]:
    """
    A solver for comparisons of paired fuzzy alternatives that uses an analysis method for evaluation and ranking

    Args:
        alternatives (List[str]): A list of lines representing alternative names
        criteria (List[str]): A list of rows representing the names of the criteria
        pairwise_matrices (List[List[List[FuzzyNumber]]]): A list of pairwise comparison matrices for each alternative
        for each criterion, containing fuzzy numbers

    Returns:
        List[Tuple[str, FuzzyNumber]]: A list of tuples, each of which contains the name of the alternative and
        the corresponding final fuzzy score, sorted in descending order

    Raises:
        ValueError: If the number of paired comparison matrices does not match the number of criteria
    """

    num_alternatives = len(alternatives)
    num_criteria = len(criteria)

    if len(pairwise_matrices) != num_criteria:
        raise ValueError("The number of paired comparison matrices must match the number of criteria")

    def normalize_matrix(matrix: List[List[FuzzyNumber]]) -> List[List[float]]:
        """
        Normalizes the transmitted matrix of fuzzy numbers

        Args:
            matrix (List[List[FuzzyNumber]]): A two-dimensional list of fuzzy numbers representing a matrix of paired comparisons

        Returns:
            List[List[float]]: A normalized matrix where each value is
            a proportion compared to the sum of the corresponding column
        """

        col_sum = [sum(col) for col in zip(*matrix)]
        normalized_matrix = [[matrix[i][j] / float(col_sum[j]) for j in range(len(matrix[i]))] for i in
                             range(len(matrix))]
        return normalized_matrix

    def get_weights(normalized_matrix: List[List[FuzzyNumber]]) -> List[FuzzyNumber]:
        """
        Calculates the weights for each row in the normalized matrix

        Args:
            normalized_matrix (List[List[FuzzyNumber]]): A normalized matrix of paired comparisons

        Returns:
            List[FuzzyNumber]: A list of weights representing the average value for each row
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
