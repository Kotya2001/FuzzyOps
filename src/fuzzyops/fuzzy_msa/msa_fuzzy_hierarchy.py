from typing import List
from fuzzyops.fuzzy_numbers import FuzzyNumber


def fuzzy_hierarchy_solver(criteria_weights: List[List[FuzzyNumber]],
                           alternative_comparisons: List[List[List[FuzzyNumber]]]) -> List[FuzzyNumber]:
    """
    A fuzzy hierarchy solver that uses a hierarchy analysis method to prioritize alternatives

    Args:
        criteria_weights (List[List[FuzzyNumber]]): A two-dimensional list of fuzzy numbers 
                representing the weights of the criteria
        alternative_comparisons (List[List[List[FuzzyNumber]]]): A three-dimensional list of fuzzy numbers representing
                comparative estimates of alternatives for each criterion

    Returns:
        List[FuzzyNumber]: A list of fuzzy numbers representing the global priorities of alternatives
    """

    def normalize_matrix(matrix: List[List[FuzzyNumber]]) -> List[float]:
        """
        Normalizes the transmitted matrix of fuzzy numbers and calculates priorities

        Args:
            matrix (List[List[FuzzyNumber]]): A two-dimensional list of fuzzy numbers representing a matrix of comparisons

        Returns:
            List[float]: A list of normalized priorities for the rows of the matrix
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
