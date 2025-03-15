from fuzzyops.fuzzy_logic import BaseRule, FuzzyInference
from fuzzyops.fuzzy_numbers import Domain

# Создание доменов для значений посроения нечетких чисел антецендентов и консеквента
service_domain = Domain((0, 10), name='service')
food_domain = Domain((0, 10), name='food')
tip_domain = Domain((0, 30), name='tip')

# Создание нечетких термов для антецендента "качество сервиса"
service_domain.create_number('gauss', 2.123, 0, name='poor') # плохой сервис
service_domain.create_number('gauss', 2.123, 5, name='good') # хороший сервис
service_domain.create_number('gauss', 2.123, 10, name='excellent') # отличный сервис

# Создание нечетких термов для антецендента "качество еды"
food_domain.create_number('trapezoidal', -9, -1, 1, 9, name='bad') # плохая еда
food_domain.create_number('trapezoidal', 7, 9, 10, 10, name='good') # хорошая еда

# Создание нечетких термов для консеквента "размер чаевых"
tip_domain.create_number('triangular', 0, 5, 10, name="cheap") # маленькие чаевые
tip_domain.create_number('triangular', 10, 15, 20, name="average") # сердние чаевые
tip_domain.create_number('triangular', 20, 25, 30, name="generous") # щедрые чаевые

# создание базы правил (консеквент всегда один, а антецедентов может быть несколько)
"""
База правил:
    Если сервис плохой и еда плохая, то чаевые маленькие;
    Если сервис средний, то чаевые средние;
    Если сервис отличный и еда вкусная, то чаевые щедрые;
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
# инициализация нечеткой системы
inference_system = FuzzyInference(domains={
    'service': service_domain,
    'food': food_domain,
    'tip': tip_domain,
}, rules=rules)

# входные данные для алгоритма
input_data = {
    'service': 3,
    'food': 10
}
# получение значения выходное переменной (дефаззифициарованное значение для чаевых)
result = inference_system.compute(input_data)
print(result)