from fuzzyops.fuzzify import triangularmf
from fuzzyops.fmath.operations import fuzzy_unite
from .utils import CudaManager
import numpy as np

if __name__ == "__main__":
    xs = np.arange(1, 10, 0.1)
    with CudaManager(xs, 4, 7, 7, func=triangularmf) as GPU:
        y = GPU.func(xs, 4, 7, 7)
        # y = triangularmf(xs, 4, 7, 7)
    print(fuzzy_unite(y, y))
