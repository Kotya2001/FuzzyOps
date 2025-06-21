"""
Task:
    It is necessary to calculate the amount of tips given in a restaurant with
    different values of the level of service and the quality of food

Fuzzy variables:
    Service - level of service (bad, medium, excellent)
    Food - quality (how much you liked the dish) of food (bad, delicious)
    Tip - Amount we want to leave the waiter (small, medium, generous)

The rule base is:
    If the service is bad and the food is bad, the tip is small;
    If the service is average, the tip is average;
    If the service is excellent and the food is delicious, the tip is generous;
"""

# (The library is already installed in your project)
from fuzzyops.fuzzy_logic import BaseRule, FuzzyInference
from fuzzyops.fuzzy_numbers import Domain

service_domain = Domain((0, 10), name='service')
food_domain = Domain((0, 10), name='food')
tip_domain = Domain((0, 30), name='tip')

service_domain.create_number('gauss', 2.123, 0, name='poor')
service_domain.create_number('gauss', 2.123, 5, name='good')
service_domain.create_number('gauss', 2.123, 10, name='excellent')

food_domain.create_number('trapezoidal', -9, -1, 1, 9, name='bad')
food_domain.create_number('trapezoidal', 7, 9, 10, 10, name='good')

tip_domain.create_number('triangular', 0, 5, 10, name="cheap")
tip_domain.create_number('triangular', 10, 15, 20, name="average")
tip_domain.create_number('triangular', 20, 25, 30, name="generous")

rules = [
    BaseRule(
        antecedents=[('service', 'poor'), ('food', 'bad')],
        consequent=('tip', 'cheap'),
    ),
    BaseRule(
        antecedents=[('service', 'good')],
        consequent=('tip', 'average'),
    ),
    BaseRule(
        antecedents=[('service', 'excellent'), ('food', 'good')],
        consequent=('tip', 'generous'),
    ),
]

inference_system = FuzzyInference(domains={
    'service': service_domain,
    'food': food_domain,
    'tip': tip_domain,
}, rules=rules)

input_data = {
    'service': 3,
    'food': 10
}

result = inference_system.compute(input_data)
print(result)