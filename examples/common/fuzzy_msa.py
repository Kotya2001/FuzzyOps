from fuzzyops.fuzzy_numbers import Domain, FuzzyNumber, memberships

from fuzzyops.fuzzy_msa import fuzzy_pareto_solver, fuzzy_sum_solver, \
    fuzzy_pairwise_solver, fuzzy_hierarchy_solver

"""
The fuzzy Pareto boundary
"""

# Creating a domain for fuzzy numbers
d = Domain((0, 101), name='d', method='minimax')

# creating fuzzy numbers for criterion 1
d.create_number('triangular', 1, 5, 11, name="x11")
d.create_number('triangular', 3, 5, 7, name='x12')
d.create_number('triangular', 0, 9, 13, name='x13')
d.create_number('triangular', 4, 5, 7, name='x14')

# creating fuzzy numbers for criterion 2
d.create_number('triangular', 3, 6, 13, name='x21')
d.create_number('triangular', 2, 7, 11, name='x22')
d.create_number('triangular', 5, 6, 7, name='x23')
d.create_number('triangular', 1, 4, 7, name='x24')

solutions = [
    [d.x11, d.x21],
    [d.x12, d.x22],
    [d.x13, d.x23],
    [d.x14, d.x24],
]

# Definition of the Pareto fuzzy boundary
pareto = fuzzy_pareto_solver(solutions)
print(pareto)


"""
Fuzzy weighted sum
"""

d.create_number('triangular', 1, 5, 11, name='weight1')
d.create_number('triangular', 3, 5, 7, name='weight2')

criteria_weights = [d.weight1, d.weight2]

# Weighted amount
weighted_sum = fuzzy_sum_solver(criteria_weights, solutions)
print("Fuzzy weighted sum:", weighted_sum)


"""
Fuzzy pairwise comparisons
"""

# Names of alternatives and criteria
alternatives = ["Product 1", "Product 2", "Product 3"]
criteria = ["Cost", "Quality"]

# Data for the pairwise comparison matrix: For each alternative and each criterion, we construct a
# pairwise comparison matrix that compares the alternatives in terms of their preference for a given criterion.
d.create_number('triangular', 1, 2, 3, name='Cost11')
d.create_number('triangular', 1, 1, 3, name='Cost12')
d.create_number('triangular', 1, 2, 5, name='Cost13')

d.create_number('triangular', 1, 2, 4, name='Cost21')
d.create_number('triangular', 1, 2, 5, name='Cost22')
d.create_number('triangular', 2, 3, 3, name='Cost23')

d.create_number('triangular', 1, 2, 3, name='Cost31')
d.create_number('triangular', 2, 2, 2, name='Cost32')
d.create_number('triangular', 1, 2, 5, name='Cost33')

d.create_number('triangular', 1, 2, 5, name='Quality11')
d.create_number('triangular', 2, 3, 4, name='Quality12')
d.create_number('triangular', 1, 2, 3, name='Quality13')

d.create_number('triangular', 1, 2, 5, name='Quality21')
d.create_number('triangular', 1, 3, 4, name='Quality22')
d.create_number('triangular', 2, 2, 3, name='Quality23')

d.create_number('triangular', 1, 3, 4, name='Quality31')
d.create_number('triangular', 2, 3, 3, name='Quality32')
d.create_number('triangular', 1, 2, 4, name='Quality33')

# matrix of pairwise evaluations of all products based on 2 criteria
pairwise_matrices = [
    [
        [d.Cost11, d.Cost12, d.Cost13],
        [d.Cost21, d.Cost22, d.Cost23],
        [d.Cost31, d.Cost32, d.Cost33],
    ],
    [
        [d.Quality11, d.Quality12, d.Quality13],
        [d.Quality21, d.Quality22, d.Quality23],
        [d.Quality31, d.Quality32, d.Quality33],
    ],
]
# Pairwise comparisons
pairwise_result = fuzzy_pairwise_solver(alternatives, criteria, pairwise_matrices)
print("Fuzzy pairwise comparisons:", pairwise_result)


"""
Fuzzy analytical hierarchy
"""

d.create_number('triangular', 1, 5, 11, name='cw11')
d.create_number('triangular', 3, 5, 7, name='cw12')
d.create_number('triangular', 0, 9, 13, name='cw13')

d.create_number('triangular', 4, 5, 7, name='cw21')
d.create_number('triangular', 3, 6, 13, name='cw22')
d.create_number('triangular', 2, 7, 11, name='cw23')

d.create_number('triangular', 5, 6, 7, name='cw31')
d.create_number('triangular', 1, 4, 7, name='cw32')
d.create_number('triangular', 2, 7, 11, name='cw33')

# Weights for each criterion
criteria_weights = [
    [d.cw11, d.cw12, d.cw13],
    [d.cw21, d.cw22, d.cw23],
    [d.cw31, d.cw32, d.cw33],
]

d.create_number('triangular', 3, 6, 13, name='cc11')
d.create_number('triangular', 3, 6, 13, name='cc12')

d.create_number('triangular', 5, 6, 7, name='cc21')
d.create_number('triangular', 2, 7, 11, name='cc22')

# Evaluation of criteria (cost criterion) for each alternative
cost_comparisons = [
    [d.cc11, d.cc12],
    [d.cc21, d.cc22],
]

d.create_number('triangular', 5, 6, 7, name='qc11')
d.create_number('triangular', 2, 7, 11, name='qc12')

d.create_number('triangular', 3, 6, 13, name='qc21')
d.create_number('triangular', 3, 6, 13, name='qc22')

# Evaluation criteria (quality criterion) for each alternative
quality_comparisons = [
    [d.qc11, d.qc12],
    [d.qc21, d.qc22],
]

d.create_number('triangular', 1, 5, 11, name='rc11')
d.create_number('triangular', 3, 5, 7, name='rc12')

d.create_number('triangular', 4, 5, 7, name='rc21')
d.create_number('triangular', 3, 6, 13, name='rc22')

# Evaluation criteria (reliability criterion) for each alternative
reliability_comparisons = [
    [d.rc11, d.rc12],
    [d.rc21, d.rc22],
]

alternative_comparisons = [cost_comparisons, quality_comparisons, reliability_comparisons]

# Analytical hierarchy
hierarchy_result = fuzzy_hierarchy_solver(criteria_weights, alternative_comparisons)
print("Fuzzy analytical hierarchy:", hierarchy_result)


