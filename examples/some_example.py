import time

import numba

import src.fuzzyops as fo

d = fo.Domain(0, 500, name='temperature (K)')
d.add_linguistic('cold', fo.fuzzify.trapezoidalmf(d.x, -1, 0, 273, 280))
d.add_linguistic('hot', fo.fuzzify.trapezoidalmf(d.x, 300, 310, 500, 500))

v0 = d.create_number('gauss', 30, 15)
v1 = d.create_number('gauss', 30, 115)
#d.plot()
d.plot()
