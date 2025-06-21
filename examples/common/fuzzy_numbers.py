from fuzzyops.fuzzy_numbers import Domain

# Creating a domain (universal set, 'minimax' minimax method)
d = Domain((1, 10, 1), name='d', method='minimax')
# Creating a domain (universal set, 'minimax' probabilistic method)
d2 = Domain((1, 10, 1), name="d2", method='prob')

# Creating fuzzy numbers in the d domain
d.create_number('triangular', 1, 3, 5, name='n1')
d.create_number('trapezoidal', 1, 3, 5, 7, name='n2')
d.create_number('gauss', 1, 5, name='n3')

# Displays a fuzzy number graph on the domain
d.plot()

# Defuzzification using the right-hand maximum method
res1 = d.n2.defuzz('rmax')
# using the center of gravity method
res2 = d.n2.defuzz()
# using the left maximum method
lmax = d.n2.defuzz('lmax')
# using the central maximum method
cmax = d.n2.defuzz('cmax')

print(res1, res2)

# Summing a fuzzy number
s = d.n2 + d.n3

# The opposite is a fuzzy number
neg = d.n1.negation

# The term "maybe"
mb = d.n2.maybe

# Display a graph of a fuzzy number
s.plot()
mb.plot()
# Gaining degrees of confidence
vals = s.values
# Obtaining the degree of confidence for a specific value from the domain (torch.Tensor)
v = s(4)
print(v.item())