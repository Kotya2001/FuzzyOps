import unittest


from src.fuzzyops.fuzzy_numbers import Domain
from src.fuzzyops.fuzzy_optimization import check_LR_type


class TestFuzzyLinearOptimization(unittest.TestCase):

    def setUp(self) -> None:
        self.d = Domain((-1, 1.5, 0.5), name='d')
        self.number = self.d.create_number('triangular', -1, 0, 1, name='n1')
        # self.number2 = self.d.create_number('trapezoidal', -1, -0.5, 0, 1, name='n2')
        # self.number3 = self.d.create_number('gauss', 1, 0, name='n3')

    def test_check_LR_type(self):
        print(check_LR_type(self.number))
        print(self.number.domain.bounds)
        return
