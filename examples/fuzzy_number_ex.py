import time
from src.fuzzyops.fuzzy_numbers import Domain, trapezoidalmf, gaussmf

d = Domain((0, 500), name='temp')
d.create_number(trapezoidalmf, -1, 0, 273, 280, name='cold')
d.create_number(trapezoidalmf, 300, 310, 500, 500, name='hot')

v0 = d.create_number(gaussmf, 30, 15, name='obs0')
v1 = d.create_number(gaussmf, 30, 115, name='obs1')
d.obs2 = v1 + 2
d.obs3 = v0 + d.obs2 + 300
d.plot()
