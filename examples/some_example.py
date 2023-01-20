import src.fuzzyops as fo
import numpy as np

xs = np.arange(1, 10, 0.1)
y = fo.fuzzify.trianglemf(xs, 4, 7, 7)
f1 = fo.FuzzyNumber(xs, y)
f1.plot()