"""
Задача:
    Необходимо рассчитать сумму выдаваемых чаевых в ресторане при
    различных значениях уровня обслуживания и качества еды

Нечеткие переменные:
    Сервис - уровень обслуживания (плохой, средний, отличный)
    Еда - качество (насколько понравилось блюдо) еды (Плохая, вкусная)
    Чаевые - Сумма, которую хотим оставить официанту (маленькие, средние, щедрые)

База правил:
    Если сервис плохой и еда плохая, то чаевые маленькие;
    Если сервис средний, то чаевые средние;
    Если сервис отличный и еда вкусная, то чаевые щедрые;
"""

# (Библиотека уже установлена в ваш проект)
from fuzzyops.fuzzy_logic import BaseRule, FuzzyInference
from fuzzyops.fuzzy_numbers import Domain, FuzzyNumber

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