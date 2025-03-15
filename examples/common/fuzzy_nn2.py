from src.fuzzyops.fuzzy_numbers import Domain
from src.fuzzyops.fuzzy_neural_net import FuzzyNNetwork

# Инициалиция модели
"""
layerSize - список, длина которого соответствует количеству слоев в нечеткой сети, а элементы, количество нейронов 
в каждом слое; 

fuzzyType - кортеж из трех элементов для задания доменной области для нечетких чисел (первое число - 
левая граница, второе число - правая граница, третье - шаг); 

method - 'minimax' - метод для нечетких вычислений ("minimax" - 
минимаксный, "prob" - вероятностный); 

fuzzyType - тип функции принадлежности, доступны следующие значения: "gauss" - гауссовское 
число, "trapezoidal" - трапецеидальное, "triangular" - треугольное (для всех нечетких чисел в данных тип должен быть 
одинаковый, поэтому задается 1 раз); 
 
activationType - тип функции активации в слоях ("linear", "relu);
"""
# Инициалиция модели
fuzzyType = 'triangular'
layerSize = [2, 2, 1]
domainVals = (-100, 100)
activationType = 'linear'
method = 'minimax'

nn = FuzzyNNetwork(
    layerSize,
    domainVals,
    method,
    fuzzyType,
    activationType
)
"""Определение необходимой мощности нагревателя (нечеткое значение больше-меньше), которое зависит от температуры 
внутри помещения и температуры снаружи помещения. 
"""
test_domain = Domain((-100, 100), name='test_domain', method='minimax')

# создание набора данных из нечетких чисел
test_domain.create_number(fuzzyType, 3, 4, 6, name='Inside1')
test_domain.create_number(fuzzyType, 2, 3, 7, name='Outside1')
test_domain.create_number(fuzzyType, -1, 3, 5, name='Power1')

test_domain.create_number(fuzzyType, 2, 3, 5, name='Inside2')
test_domain.create_number(fuzzyType, 3, 5, 6, name='Outside2')
test_domain.create_number(fuzzyType, 0, 2, 4, name='Power2')

test_domain.create_number(fuzzyType, 0, 1, 2, name='Inside3')
test_domain.create_number(fuzzyType, -1, 0, 1, name='Outside3')
test_domain.create_number(fuzzyType, -1, 0, 3, name='Power3')

# Создаем данные для модели
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