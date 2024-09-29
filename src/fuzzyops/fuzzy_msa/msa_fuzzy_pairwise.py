
def fuzzy_pairwise_solver(alternatives, criteria, pairwise_matrices):
    num_alternatives = len(alternatives)
    num_criteria = len(criteria)

    if len(pairwise_matrices) != num_criteria:
        raise ValueError("Число матриц парных сравнений должно совпадать с числом критериев.")

    def normalize_matrix(matrix):
        col_sum = [sum(col) for col in zip(*matrix)]
        normalized_matrix = [[matrix[i][j] / float(col_sum[j]) for j in range(len(matrix[i]))] for i in range(len(matrix))]
        return normalized_matrix

    def get_weights(normalized_matrix):
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
