import unittest
import numpy as np
from ..fuzzify import Domain


class TestFuzzyNumber(unittest.TestCase):
    """
    Test functionality of FuzzyNumber class
    """

    def setUp(self) -> None:
        self.d = Domain((-1, 1.5, 0.5), name='d')
        self.d.create_number('triangular', -1, 0, 1, name='n1')
        self.d.create_number('trapezoidal', -1, -0.5, 0, 1, name='n2')
        self.d.create_number('gauss', 1, 0, name='n3')

    def test_fuzz(self) -> None:
        """
        Тест на создания нечеткого числа(треугольное, трапецеидальное, гауссовское)
        """
        self.assertTrue(hasattr(self.d, 'n1'), 'Fuzzy number is not an attribute of domain')

        # test triangular
        membership = np.array([0, 0.5, 1, 0.5, 0])
        self.assertTrue(np.allclose(membership, self.d.n1.values), 'Triangular membership is not correct')

        # test trapezoidal
        membership = np.array([0, 1, 1, 0.5, 0])
        self.assertTrue(np.allclose(membership, self.d.n2.values), 'Trapezoidal membership is not correct')

        # test gaussian
        membership = np.array([np.exp(-0.5), np.exp(-0.125), 1, np.exp(-0.125), np.exp(-0.5)])
        self.assertTrue(np.allclose(membership, self.d.n3.values), 'Gaussian membership is not correct')

    def test_defuzz(self) -> None:
        """
        Тест на проверку операций дефаззицикайции
        """
        # test lmax
        self.assertAlmostEqual(-0.5, self.d.n2.defuzz('lmax'), 6, 'Left max is not correct')

        # test rmax
        self.assertAlmostEqual(0, self.d.n2.defuzz('rmax'), 6, 'Right max is not correct')

        # test cmax
        self.assertAlmostEqual(-0.25, self.d.n2.defuzz('cmax'), 6, 'Center of maximums is not correct')

        # test cgrav
        self.assertAlmostEqual(-0.1, self.d.n2.defuzz('cgrav'), 6, 'Center of gravity is not correct')

    def test_operations(self) -> None:
        """
        Тест на провреку арифсетических операций между нечеткими числами)
        """
        # test very
        membership = np.array([0, 0.25, 1, 0.25, 0])
        self.assertTrue(np.allclose(membership, self.d.n1.very.values), 'Operation "very" is not correct')

        # test maybe
        membership = np.array([0, 0.5 ** 0.5, 1, 0.5 ** 0.5, 0])
        self.assertTrue(np.allclose(membership, self.d.n1.maybe.values), 'Operation "maybe" is not correct')

        # test neg
        membership = np.array([1, 0.5, 0, 0.5, 1])
        self.assertTrue(np.allclose(membership, self.d.n1.negation.values), 'Operation "negation" is not correct')

        # minimax operations
        # test add
        membership = np.array([0, 1, 1, 0.5, 0])
        self.assertTrue(np.allclose(membership, (self.d.n1 + self.d.n2).values),
                        'Operation "minimax add" is not correct')

        # test sub
        membership = np.array([0, 0.5, 0, 0, 0])
        self.assertTrue(np.allclose(membership, (self.d.n2 - self.d.n1).values),
                        'Operation "minimax sub" is not correct')

        # test mul
        membership = np.array([0, 0.5, 1, 0.5, 0])
        self.assertTrue(np.allclose(membership, (self.d.n1 * self.d.n2).values),
                        'Operation "minimax mul" is not correct')

        # prob operations
        self.d.method = 'prob'
        # test add
        membership = np.array([0, 1, 1, 0.75, 0])
        self.assertTrue(np.allclose(membership, (self.d.n1 + self.d.n2).values),
                        'Operation "prob add" is not correct')

        # test mul
        membership = np.array([0, 0.5, 1, 0.25, 0])
        self.assertTrue(np.allclose(membership, (self.d.n1 * self.d.n2).values),
                        'Operation "prob mul" is not correct')

        # operations with scalars
        # test add
        membership = np.array([0, 0, 0, 0.5, 1])
        self.assertTrue(np.allclose(membership, (self.d.n1 + 1).values),
                        'Operation "add scalar" is not correct')

        # test sub
        membership = np.array([1, 0.5, 0, 0, 0])
        self.assertTrue(np.allclose(membership, (self.d.n1 - 1).values),
                        'Operation "sub scalar" is not correct')

    @unittest.skip('Not implemented yet')
    def test_plot(self) -> None:
        """
        Тест на отрисовку графика
        """
        # test plot
        self.d.n1.plot()

    def test_str(self) -> None:
        self.assertEqual('-0.1', str(self.d.n2)[:4], 'String representation is not correct')

    def test_repr(self) -> None:
        self.assertEqual('Fuzzy-0.1', repr(self.d.n2)[:9], 'String representation is not correct')

    def test_functions(self) -> None:
        """
        Проверка альфа-среза/коэффциента энтропии
        """
        # test alpha-cut
        vals = [-0.5, 0, 0.5]
        self.assertTrue(np.allclose(vals, self.d.n2.alpha_cut(0.5)), 'Alpha-cut is not correct')

        # test entropy
        self.assertEqual(0.2, self.d.n2.entropy(), 'Entropy is not correct')
