from fuzzyops.fuzzy_numbers import Domain

# Создание домена (универсального множества, 'minimax' метод минимаксный)
d = Domain((1, 10, 1), name='d', method='minimax')
# Создание домена (универсального множества, 'minimax' метод вероятностный)
d2 = Domain((1, 10, 1), name="d2", method='prob')

# Создание нечетких чисел на домене d
d.create_number('triangular', 1, 3, 5, name='n1')
d.create_number('trapezoidal', 1, 3, 5, 7, name='n2')
d.create_number('gauss', 1, 5, name='n3')

# Отобразит график нечеткого числа на домене
d.plot()

# Дефаззификаци, методом правого максимума
res1 = d.n2.defuzz('rmax')
# методом центра тяжести
res2 = d.n2.defuzz()
# методом левого максимума
lmax = d.n2.defuzz('lmax')
# методом центрального максимума
cmax = d.n2.defuzz('cmax')

print(res1, res2)

# Суммирование нечеткого числа
s = d.n2 + d.n3

# Противоположное нечеткое число
neg = d.n1.negation

# Терм "возможно"
mb = d.n2.maybe

# Отобразить гарафик нечеткого числа
s.plot()
mb.plot()
# Получение степеней уверенности
vals = s.values
# Получение степени уверенности для конкретного значения из домена (torch.Tensor)
v = s(4)
print(v.item())