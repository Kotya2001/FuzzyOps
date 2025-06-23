from fuzzyops.fuzzy_numbers import Domain
from fuzzyops.fuzzy_msa import fuzzy_pareto_solver, fuzzy_sum_solver, fuzzy_pairwise_solver, fuzzy_hierarchy_solver

"""
Task:
Choosing a supplier to purchase equipment

The company chooses from four equipment suppliers, taking into account two criteria: price (cost) – the lower, the better and
quality (quality) – the higher, the better.

Each supplier offers different conditions (price and quality), presented in the form of fuzzy numbers reflecting
the uncertainty in the assessment.

The Pareto boundary method allows you to find a set of optimal suppliers where it is impossible to improve one
indicator (for example, quality) without worsening the other (price). This method helps logisticians and purchasers select
the best suppliers without explicitly weighing the criteria.

In addition, you can determine the importance of each parameter through a fuzzy weight factor. Then, having these weights,
we can determine the rating of each of the alternatives by finding the optimal one using a fuzzy weighted sum. 
The method is useful when choosing among complex alternatives with several characteristics.

"""

d = Domain((-100, 1000), name='d', method='minimax')

######################################################
# 1. Defining the Pareto boundary
######################################################


# Criteria: Price and quality
d.create_number('triangular', 1, 5, 11, name='cost1')
d.create_number('triangular', 3, 5, 7, name='cost2')
d.create_number('triangular', 0, 9, 13, name='cost3')
d.create_number('triangular', 4, 5, 7, name='cost4')
d.create_number('triangular', 3, 6, 13, name='quality1')
d.create_number('triangular', 2, 7, 11, name='quality2')
d.create_number('triangular', 5, 6, 7, name='quality3')
d.create_number('triangular', 1, 4, 7, name='quality4')

# Alternatives: 4 suppliers
alternatives_scores = [
    [d.cost1, d.quality1],
    [d.cost2, d.quality2],
    [d.cost3, d.quality3],
    [d.cost4, d.quality4],
]

# The Pareto boundary
pareto_front = fuzzy_pareto_solver(alternatives_scores)
print("The fuzzy Pareto boundary:", pareto_front)

######################################################
# 2. Definition of a fuzzy weighted sum
######################################################

d.create_number('triangular', 1, 5, 11, name='weight1')
d.create_number('triangular', 3, 5, 7, name='weight2')

criteria_weights = [d.weight1, d.weight2]

# Weighted amount
weighted_sum = fuzzy_sum_solver(criteria_weights, alternatives_scores)
print("Fuzzy weighted sum:", weighted_sum)

"""

Task:
Choosing the best equipment option among several models

The decision-maker chooses between three models of industrial equipment for the company, taking into account the cost (cheaper is better) and 
quality (better quality is better ). 

 All smartphones are compared pairwise in terms of cost and quality, using fuzzy assessments 
(for example, "slightly better", "much better", etc.). The pairwise comparison method allows you to rank the options, 
identifying the best one based on all the criteria.

"""

######################################################
# 3. Определение нечетких попарных сравнений
######################################################


alternatives = ["Equipment 1", "Equipment 2", "Equipment 3"]
criteria = ["Cost", "Quality"]

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
# Попарные сравнения
pairwise_result = fuzzy_pairwise_solver(alternatives, criteria, pairwise_matrices)
print("Fuzzy pairwise comparisons:", pairwise_result)

"""
Task: 
Choosing the best candidate for the position of manager

The HR department of the company analyzes candidates for the position of manager, evaluating them according to three main criteria cost of hiring 
(salary expectations, training costs), quality of work (experience, skills), reliability (level of responsibility, 
recommendations). Each criterion is broken down is collective, for example, "quality of work" takes into account both experience and 
leadership qualities.

The analytic hierarchy process allows you to divide a complex decision into levels, compare candidates on all parameters and 
choose the best. Used in recruitment, strategic planning and investment decisions.

"""

######################################################
# 4. Defining a fuzzy hierarchy
######################################################

d.create_number('triangular', 1, 5, 11, name='cw11')
d.create_number('triangular', 3, 5, 7, name='cw12')
d.create_number('triangular', 0, 9, 13, name='cw13')

d.create_number('triangular', 4, 5, 7, name='cw21')
d.create_number('triangular', 3, 6, 13, name='cw22')
d.create_number('triangular', 2, 7, 11, name='cw23')

d.create_number('triangular', 5, 6, 7, name='cw31')
d.create_number('triangular', 1, 4, 7, name='cw32')
d.create_number('triangular', 2, 7, 11, name='cw33')

criteria_weights = [
    [d.cw11, d.cw12, d.cw13],
    [d.cw21, d.cw22, d.cw23],
    [d.cw31, d.cw32, d.cw33],
]

d.create_number('triangular', 3, 6, 13, name='cc11')
d.create_number('triangular', 3, 6, 13, name='cc12')

d.create_number('triangular', 5, 6, 7, name='cc21')
d.create_number('triangular', 2, 7, 11, name='cc22')

cost_comparisons = [
    [d.cc11, d.cc12],
    [d.cc21, d.cc22],
]

d.create_number('triangular', 5, 6, 7, name='qc11')
d.create_number('triangular', 2, 7, 11, name='qc12')

d.create_number('triangular', 3, 6, 13, name='qc21')
d.create_number('triangular', 3, 6, 13, name='qc22')

quality_comparisons = [
    [d.qc11, d.q12],
    [d.qc21, d.qc22],
]

d.create_number('triangular', 1, 5, 11, name='rc11')
d.create_number('triangular', 3, 5, 7, name='rc12')

d.create_number('triangular', 4, 5, 7, name='rc21')
d.create_number('triangular', 3, 6, 13, name='rc22')

reliability_comparisons = [
    [d.rc11, d.rc12],
    [d.rc21, d.rc22],
]

alternative_comparisons = [cost_comparisons, quality_comparisons, reliability_comparisons]

# Analytical hierarchy
hierarchy_result = fuzzy_hierarchy_solver(criteria_weights, alternative_comparisons)
print("Fuzzy analytical hierarchy:", hierarchy_result)
