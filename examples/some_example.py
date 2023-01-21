import numba

import src.fuzzyops as fo
import numpy as np
from time import perf_counter
from numba import cuda
print(cuda.gpus)
import math

cuda.select_device(0)

@numba.njit
def unite_all_sets(args):
    mins = []
    steps = []
    maxs = []
    for num in args:
        x = num.get_x()
        mins.append(x[0])
        steps.append(x[1] - x[0])
        maxs.append(x[-1])
    mi = np.min(mins)
    ma = np.max(maxs)
    step = np.min(steps)
    return np.arange(mi, ma + step, step)


def get_all_xs(fs):
    for f in fs:
        yield f.get_x()


def get_all_vals(fs):
    for f in fs:
        yield f.get_x()

@numba.vectorize(target='cuda')
def min_of_0(a, b):
    return min(a[0], b[0])

@numba.vectorize(target='cuda')
def min_of_neg1(a, b):
    return min(a[-1], b[-1])

@numba.vectorize(target='cuda')
def min_of_diffs(a, b):
    return min(a[1] - a[0], b[1] - b[0])


def extend_all_vals(x, *args):
    for arg in args:
       yield arg.extend_values(x)

@numba.njit
def sum_all(xs, ys):
    x = unite_all_sets(xs)
    vals = np.array(extend_all_vals(ys, x))
    result = vals[0]
    for val in vals[1:]:
        result += val
    return result

'''
xs = np.arange(1, 10, 0.1)
y = fo.fuzzify.trianglemf(xs, 4, 7, 7).astype('float32')
f1 = fo.FuzzyNumber(xs, y)
xs2 = np.arange(2, 15, 0.2)
y2 = fo.fuzzify.trapezoidalmf(xs2, 3, 5, 7, 10).astype('float32')
f2 = fo.FuzzyNumber(xs2, y2)
f3 = f1 + f2 - f1
st = perf_counter()
#sum_all([f1, f2, f3])
en = perf_counter()
print(en-st)

#f1.plot()
#f2.plot()
#f3.plot()
print(list(get_all_xs([f1, f2, f3])))'''

d = fo.Domain(-100, 100, name='temperature')  # , method='prob')
d.add_linguistic('cold', fo.fuzzify.trapezoidalmf(d.x, -100, -100, 0, 10))
d.add_linguistic('hot', fo.fuzzify.trapezoidalmf(d.x, 30, 50, 100, 100))
f1 = d.create_number('triangular', 50, 60, 80)
f2 = d.create_number('triangular', 20, 40, 55)
f3 = d.create_number('trapezoidal', 10, 25, 40, 50)
st = perf_counter()
for i in range(1000):
    ft = f3 + f1
    #ftv = np.maximum(f3.values, f1.values)
en = perf_counter()
print(en-st)
