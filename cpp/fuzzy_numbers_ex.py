from fuzzyops.fuzzy_numbers import Domain

d = Domain((0, 101), name='d', method='minimax')
d.create_number('gauss', 1, 0, name='out')
for i in range(50):
    d.create_number('gauss', 1, i, name='n' + str(i))
    d.out += d.get('n' + str(i))

d.to('cpu')
print(d.out)
print(d.out.values)