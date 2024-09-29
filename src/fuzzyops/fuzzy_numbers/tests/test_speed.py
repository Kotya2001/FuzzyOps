import unittest
from time import perf_counter
from ..fuzzify import Domain
import sys


class TestSpeed(unittest.TestCase):

    def setUp(self) -> None:
        """
        Инициализация теста на скорость
        """
        sys.setrecursionlimit(1500)
        self.d = Domain((0, 101), name='d', method='minimax')
        self.d.create_number('gauss', 1, 0, name='out')
        for i in range(1000):
            self.d.create_number('gauss', 1, i // 10, name='n' + str(i))
            self.d.out += self.d.get('n' + str(i))

        self.m = Domain((0, 101), name='m', method='minimax')
        self.m.create_number('gauss', 1, 0, name='mul')
        for i in range(1000):
            self.m.create_number('gauss', 1, i // 10, name='n' + str(i) + 'mul')
            self.m.mul *= self.m.get('n' + str(i) + 'mul')

    def test_speed_cpu_minimax_add(self) -> None:
        """
        Теста на скорость операции сложения по медоту minmax между нечеткими числами, на обычном процессоре
        """
        # test speed of operations on 1000 fuzzy numbers with 100 segments on numpy
        self.d.to('cpu')
        self.d.method = 'minimax'
        start = perf_counter()
        values = self.d.out.values
        end = perf_counter()
        print('cpu minimax:', end - start)
        self.assertLess(end - start, 5, 'Speed of operations on numpy is too slow')

    def test_speed_cpu_prob_add(self) -> None:
        """
        Теста на скорость операций по вероятностному методу между нечеткими числами, на обычном процессоре
        """
        # test speed of operations on 1000 fuzzy numbers with 100 segments on numpy
        self.d.to('cpu')
        self.d.method = 'prob'
        start = perf_counter()
        values = self.d.out.values
        end = perf_counter()
        print('cpu prob:', end - start)
        self.assertLess(end - start, 5, 'Speed of operations on numpy is too slow')

    def test_speed_cpu_minimax_mul(self) -> None:
        """
        Теста на скорость операции умножения по медоту minmax между нечеткими числами, на обычном процессоре
        """
        # test speed of operations on 1000 fuzzy numbers with 100 segments on numpy
        self.m.to('cpu')
        self.m.method = 'minimax'
        start = perf_counter()
        values = self.m.mul.values
        end = perf_counter()
        print('cpu minimax:', end - start)
        self.assertLess(end - start, 5, 'Speed of operations on numpy is too slow')

    def test_speed_cpu_prob_mul(self) -> None:
        """
        Теста на скорость операции умножение по вероятностному методу между нечеткими числами, на обычном процессоре
        """
        # test speed of operations on 1000 fuzzy numbers with 100 segments on numpy
        self.m.to('cpu')
        self.m.method = 'prob'
        start = perf_counter()
        values = self.m.mul.values
        end = perf_counter()
        print('cpu prob:', end - start)
        self.assertLess(end - start, 5, 'Speed of operations on numpy is too slow')

    def test_speed_cuda_minimax_add(self) -> None:
        """
        Теста на скорость операции сложения по minimax методу между нечеткими числами, на графическом процессоре
        """
        # test speed of operations on 1000 fuzzy numbers with 100 segments on cuda
        self.d.to('cuda')
        self.d.method = 'minimax'
        start = perf_counter()
        values = self.d.out.values
        end = perf_counter()
        print('cuda minimax:', end - start)
        self.assertLess(end - start, 5, 'Speed of operations on cuda is too slow')

    def test_speed_cuda_prob_add(self) -> None:
        """
        Теста на скорость операции сложения по вероятностному методу между нечеткими числами, на графическом процессоре
        """
        # test speed of operations on 1000 fuzzy numbers with 100 segments on cuda
        self.d.to('cuda')
        self.d.method = 'prob'
        start = perf_counter()
        values = self.d.out.values
        end = perf_counter()
        print('cuda prob:', end - start)
        self.assertLess(end - start, 5, 'Speed of operations on cuda is too slow')

    def test_speed_cuda_minimax_mul(self) -> None:
        """
        Теста на скорость операции умножения по minimax методу между нечеткими числами, на графическом процессоре
        """
        # test speed of operations on 1000 fuzzy numbers with 100 segments on cuda
        self.m.to('cuda')
        self.m.method = 'minimax'
        start = perf_counter()
        values = self.m.mul.values
        end = perf_counter()
        print('cuda minimax:', end - start)
        self.assertLess(end - start, 5, 'Speed of operations on cuda is too slow')

    def test_speed_cuda_prob_mul(self) -> None:
        """
        Теста на скорость операции умножения по вероятностному методу между нечеткими числами, на графическом процессоре
        """
        # test speed of operations on 1000 fuzzy numbers with 100 segments on cuda
        self.m.to('cuda')
        self.m.method = 'prob'
        start = perf_counter()
        values = self.m.mul.values
        end = perf_counter()
        print('cuda prob:', end - start)
        self.assertLess(end - start, 5, 'Speed of operations on cuda is too slow')
