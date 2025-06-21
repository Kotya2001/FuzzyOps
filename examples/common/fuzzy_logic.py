from fuzzyops.fuzzy_logic import BaseRule, FuzzyInference
from fuzzyops.fuzzy_numbers import Domain

# Creating domains for the values of the construction of fuzzy numbers of antecedents and consequences
service_domain = Domain((0, 10), name='service')
food_domain = Domain((0, 10), name='food')
tip_domain = Domain((0, 30), name='tip')

# Creating fuzzy terms for the antecedent "service quality"
service_domain.create_number('gauss', 2.123, 0, name='poor') # poor service
service_domain.create_number('gauss', 2.123, 5, name='good') # хороший сервис
service_domain.create_number('gauss', 2.123, 10, name='excellent') # excellent service

# Creating fuzzy terms for the "food quality" antecedent
food_domain.create_number('trapezoidal', -9, -1, 1, 9, name='bad') # bad food
food_domain.create_number('trapezoidal', 7, 9, 10, 10, name='good') # good food

# Creating fuzzy terms for the "tip size" consequent
tip_domain.create_number('triangular', 0, 5, 10, name="cheap") # cheap tips
tip_domain.create_number('triangular', 10, 15, 20, name="average") # average tips
tip_domain.create_number('triangular', 20, 25, 30, name="generous") # generous tips

# creating a rule base (there is always one consequence, and there may be several antecedents)
"""
The rule base is:
 If the service is bad and the food is bad, the tip is small;
 If the service is average, the tip is average;
 If the service is excellent and the food is delicious, the tip is generous;
"""
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
# initializing a fuzzy system
inference_system = FuzzyInference(domains={
    'service': service_domain,
    'food': food_domain,
    'tip': tip_domain,
}, rules=rules)

# input data for the algorithm
input_data = {
    'service': 3,
    'food': 10
}
# obtaining the value of the output variable (the de-fuzzified value for tips)
result = inference_system.compute(input_data)
print(result)