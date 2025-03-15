import numpy as np
from fuzzyops.fuzzy_numbers import Domain
from fuzzyops.fuzzy_logic import BaseRule, SingletonInference


# генерируем случаынйе параметры для треугольных чисел
def generate_params(n, low, high):
    p = np.zeros((n, 3))
    for j in range(n):
        p[j, :] = np.random.uniform(low=low, high=high, size=(3,))
    return p


# Создание тестового набора данных
x = np.arange(start=0.01, stop=1, step=0.01)
# Значения целевой переменной
r = np.array([9.919, -6.175, 4.372, -3.680, 2.663, -2.227,
              1.742, -2.789, 11.851, -8.565, 0.938, -0.103])
size = r.shape[0]

# В выборке 1 признак и 1 целевая переменная, необходимо найти
X = np.random.choice(x, size=size)
X = np.reshape(X, (size, 1))
data = np.hstack((X, np.reshape(r, (size, 1))))

low, high = np.min(data[:, 0]), np.max(data[:, 0])
# создадим случайные параметры для треугольной функции принадлежности
params = generate_params(size, low, high)
params.sort()

# Строим доменную область для входной переменной
x = Domain((np.min(data[:, 0]) - 2, np.max(data[:, 0]), 0.1), name='x')
for i in range(data.shape[0]):
    x.create_number('triangular', *params[i, :].tolist(), name=f"x_{i}")

# строим базу правил типа Синглтон (значение консеквента одно и это четкое число)
rul = [
    BaseRule(antecedents=[('x', f'x_{i}')], consequent=data[:, -1][i])
    for i in range(data.shape[0])
]

inference_system = SingletonInference(domains={
    'x': x,
}, rules=rul)

# Подаем данные на вход получаем результат
input_data = {'x': 0.66}
result = inference_system.compute(input_data)
print(result)