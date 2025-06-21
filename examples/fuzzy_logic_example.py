"""
Task:
    It is necessary to simulate the probability of an attacker's intrusion into a corporate system, as well as
    the probability of an attack spreading by means of building a rule base and fuzzy inference

Fuzzy variables:
    X1 - Number of active users in the system
    X2 - Time (hours)
    Y1 - Possibility of penetration
    Y2 - Possibility of attack propagation

Rule base:
    If users (X1) are "several" and time (X2) is "non-working", then the possibility of penetration (Y1) is "medium";
    If users (X1) are "many" and time (X2) is "working", then the possibility of penetration (Y1) is "low";
    If users (X1) are "many" and time (X2) is "non-working", then the possibility of penetration (Y1) is "high";
    
    If the number of users (X1) is "few" and the time (X2) is "non-working", then the possibility of an attack spreading (Y2) is "medium";
    If the number of users (X1) is "many" and the time (X2) is "working", then the possibility of an attack spreading (Y2) is "low";
    If the number of users (X1) is "many" and the time (X2) is "few", then the possibility of an attack spreading (Y2) is "high";
"""

# Importing the necessary classes for constructing fuzzy logical inference using the Mamdini
# algorithm (The library is already installed in your project)
from fuzzyops.fuzzy_logic import BaseRule, FuzzyInference
from fuzzyops.fuzzy_numbers import Domain, FuzzyNumber


# The rule base
rules = [
    BaseRule(
        antecedents=[('user', 'several'), ('time', 'not_work_time')],
        consequent=('attack', 'middle'),
    ),
    BaseRule(
        antecedents=[('user', 'many'), ('time', 'work_time')],
        consequent=('attack', 'low'),
    ),
    BaseRule(
        antecedents=[('user', 'many'), ('time', 'not_work_time')],
        consequent=('attack', 'high'),
    ),
    BaseRule(
        antecedents=[('user', 'several'), ('time', 'not_work_time')],
        consequent=('spread', 'middle'),
    ),
    BaseRule(
        antecedents=[('user', 'many'), ('time', 'work_time')],
        consequent=('spread', 'low'),
    ),
    BaseRule(
        antecedents=[('user', 'many'), ('time', 'not_work_time')],
        consequent=('spread', 'high'),
    ),
]

# X1
user_domain = Domain((0, 30), name='user')
user_domain.create_number('trapezoidal', -1, 0, 4, 7, name='several')
# creating the term "many" users
many_users = user_domain.get('several').negation
user_domain.many = many_users

# X2
time_domain = Domain((0, 24), name='time')
time_domain.create_number('trapezoidal', 8, 9, 18, 19, name='work_time')
# # creating a term for "non-working" time
not_work_time = time_domain.get('work_time').negation
time_domain.not_work_time = not_work_time

# Y1
# Fuzzy terms for attack probabilities
attack_prob = Domain((0, 1, 0.01), name='attack')
attack_prob.create_number('trapezoidal', -0.1, 0., 0.1, 0.3, name='low')
attack_prob.create_number('trapezoidal', 0.3, 0.43, 0.6, 0.7, name='middle')
attack_prob.create_number('trapezoidal', 0.65, 0.8, 0.9, 1, name='high')

# Y2
# Fuzzy terms for propagation probabilities
spread_prob = Domain((0, 1, 0.01), name='spread')
spread_prob.create_number('trapezoidal', -0.1, 0., 0.15, 0.34, name='low')
spread_prob.create_number('trapezoidal', 0.32, 0.4, 0.5, 0.7, name='middle')
spread_prob.create_number('trapezoidal', 0.64, 0.76, 0.9, 1, name='high')


inference_system = FuzzyInference(domains={
    'user': user_domain,
    'time': time_domain,
    'attack': attack_prob,
    'spread': spread_prob,
}, rules=rules)

input_data = {
    'user': 35,  # A lot of people are still at work
    'time': 17  # 17:00
}

result = inference_system.compute(input_data)
print(result)
