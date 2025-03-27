from src.fuzzyops.fuzzy_numbers import Domain
from src.fuzzyops.fuzzy_neural_net import FuzzyNNetwork


"""
Задача:
Разрабатывается система для управления энергопотреблением кондиционеров на заводе. Нужно предсказать, сколько мощности
потребуется кондиционерам в зависимости от внутренней температуры помещения (Inside) и внешней температуры окружающей 
среды (Outside).

Из-за неопределённости в данных (например, колебания температуры, ошибки измерений) следует использовать нечеткую 
нейронную сеть, которая работает с треугольными нечеткими числами. Данная сеть обучается на тестовых входных данных,
а после этого ее можно применять для расчетов в реальной технике.

"""


fuzzyType = 'triangular'
nn = FuzzyNNetwork(
    [2, 2, 1],
    (-100, 100),
    'minimax',
    fuzzyType,
    'linear'
)

test_domain = Domain((-100, 100), name='test_domain', method='minimax')

# создание тестового набора данных

test_domain.create_number(fuzzyType, 3, 4, 6, name='Inside1')
test_domain.create_number(fuzzyType, 2, 3, 7, name='Outside1')
test_domain.create_number(fuzzyType, -1, 3, 5, name='Power1')

test_domain.create_number(fuzzyType, 2, 3, 5, name='Inside2')
test_domain.create_number(fuzzyType, 3, 5, 6, name='Outside2')
test_domain.create_number(fuzzyType, 0, 2, 4, name='Power2')

test_domain.create_number(fuzzyType, 0, 1, 2, name='Inside3')
test_domain.create_number(fuzzyType, -1, 0, 1, name='Outside3')
test_domain.create_number(fuzzyType, -1, 0, 3, name='Power3')


X_train = [
    [test_domain.Inside1, test_domain.Outside1],
    [test_domain.Inside2, test_domain.Outside2],
    [test_domain.Inside3, test_domain.Outside3],
]

y_train = [
    [test_domain.Power1],
    [test_domain.Power2],
    [test_domain.Power3],
]

test_domain.create_number(fuzzyType, -1, 0, 1, name='Inside')
test_domain.create_number(fuzzyType, -1, 0, 1, name='Outside')

X_test = [test_domain.Inside, test_domain.Outside]

nn.fit(X_train, y_train, 100)

result = nn.predict(X_test)

print(result)