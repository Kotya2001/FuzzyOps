from .mf import (trianglemf, trapezoidalmf)
from src.fuzzyops._fuzzynumber import FuzzyNumber


# DEPRECIATED
# def fuzzify(*x, mf=trianglemf, method='minimax'):
#    assert method == 'minimax' or method == 'prob', "Unknown method. Known methods are 'minimax' and 'prob'"
#    y = mf(*x)
#    return FuzzyNumber(x, y, method)
