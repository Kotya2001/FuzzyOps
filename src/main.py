from fuzzyops.fuzzify import trianglemf
from fuzzyops.math.operations import fuzzy_unite
import numpy as np

if __name__ == "__main__":
    xs = np.arange(1, 10, 0.1)
    y = trianglemf(xs, 4, 7, 7)
    print(fuzzy_unite(y, y))
