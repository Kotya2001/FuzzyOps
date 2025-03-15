from fuzzyops.fuzzy_logic import BaseRule, FuzzyInference
from fuzzyops.fuzzy_numbers import Domain

age_domain = Domain((0, 100), name='age')
age_domain.create_number('trapezoidal', -1, 0, 20, 30, name='young')
age_domain.create_number('trapezoidal', 20, 30, 50, 60, name='middle')
age_domain.create_number('trapezoidal', 50, 60, 100, 100, name='old')

accident_domain = Domain((0, 1, 0.1), name='accident')
accident_domain.create_number('trapezoidal', -0.1, 0., 0.1, 0.2, name='low')
accident_domain.create_number('trapezoidal', 0.1, 0.2, 0.7, 0.8, name='medium')
accident_domain.create_number('trapezoidal', 0.7, 0.8, 0.9, 1, name='high')

ruleset = [
            BaseRule(
                antecedents=[('age', 'young')],
                consequent=('accident', 'high'),
            ),
            BaseRule(
                antecedents=[('age', 'middle')],
                consequent=('accident', 'medium'),
            ),
            BaseRule(
                antecedents=[('age', 'old')],
                consequent=('accident', 'high'),
            ),
        ]

fuzzy_inference = FuzzyInference(domains={
            'age': age_domain,
            'accident': accident_domain,
        }, rules=ruleset)

minimax_out = fuzzy_inference.compute({"age": 25})
print(minimax_out['accident'])