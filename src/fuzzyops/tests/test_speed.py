import unittest
from time import perf_counter

import sys
import os
from pathlib import Path

root_path = Path(os.path.abspath(__file__))
src_dir = root_path.parents[2]
sys.path.append(src_dir.__str__())

from fuzzyops.fuzzy_numbers import Domain


class TestSpeed(unittest.TestCase):

    def setUp(self) -> None:
        """
        Initializing the speed test

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
        A test for the speed of the minmax addition operation between fuzzy numbers, on a conventional processor

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
        A test for the speed of the addition operation using the probabilistic method between fuzzy numbers, on a conventional processor

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
        A test for the speed of the minmax multiplication operation between fuzzy numbers, on a conventional processor

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
        A test for the speed of the multiplication operation using the probabilistic method between fuzzy numbers, on a conventional processor

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
        A test for the speed of the minimax addition operation between fuzzy numbers on a GPU

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
        A test for the speed of the addition operation using the probabilistic method between fuzzy numbers, on a GPU

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
        A test for the speed of the minimax multiplication operation between fuzzy numbers on a GPU

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
        A test for the speed of the multiplication operation using the probabilistic method between fuzzy numbers, on a GPU
        
        """
        # test speed of operations on 1000 fuzzy numbers with 100 segments on cuda
        self.m.to('cuda')
        self.m.method = 'prob'
        start = perf_counter()
        values = self.m.mul.values
        end = perf_counter()
        print('cuda prob:', end - start)
        self.assertLess(end - start, 5, 'Speed of operations on cuda is too slow')
