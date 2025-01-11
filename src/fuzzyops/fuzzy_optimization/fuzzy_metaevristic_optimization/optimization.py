

from fuzzyops.fuzzy_numbers import Domain, FuzzyNumber

from typing import Union
import numpy as np
import pandas as pd
from scipy import interpolate
from dataclasses import dataclass

tps = Union[FuzzyNumber, float]

params = {"triangular": 3, "trapezoidal": 4, "gauss": 2, "bell": 3}


def __gaussian_f(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Генерирует нормальные случайные числа на основе заданного среднего и стандартного отклонения.

    Args:
        mu (np.ndarray): Среднее значение для нормального распределения.
        sigma (np.ndarray): Стандартное отклонение для нормального распределения.

    Returns:
        np.ndarray: Сгенерированные нормальные случайные числа.
    """

    return np.random.default_rng().normal(loc=mu, scale=sigma)


vector_gaussian_f = np.vectorize(__gaussian_f)


@dataclass(frozen=True)
class FuzzyBounds:
    """
    Определяет границы для нечетких значений.

    Attributes:
        start (Union[int, float]): Начальное значение границ.
        end (Union[int, float]): Конечное значение границ.
        step (Union[int, float]): Шаг для границ.
        x (list[str]): Список меток или названий границ.
    """

    start: Union[int, float]
    end: Union[int, float]
    step: Union[int, float]
    x: str
    # x: list[str]


@dataclass
class Archive:
    """
    Хранит параметры и результаты для одной итерации оптимизации.

    Attributes:
        k (int): Индекс архива.
        params (np.ndarray): Параметры нечеткой модели.
        loss (float): Потери (ошибка) для данной итерации.
    """

    k: int
    params: np.ndarray
    loss: float


class AntOptimization:
    """
    Алгоритм оптимизации муравьиных колоний для идентификации нечетких систем.
    Алгоритм реаоизован по статье:
    И.А. Ходашинский, П.А. Дудин. Идентификация нечетких систем на основе непрерывного алгоритма муравьиных колоний.
    Автометрия. 2012. Т. 48, № 1.

    Args:
        data (np.ndarray): Входные данные для создания модели.
        k (int): Количество начальных решений.
        epsilon (float): Параметр для изменения веса.
        q (float): Параметр для нормализации потерь.
        n_iter (int): Количество итераций алгоритма.
        n_ant (int): Общее количество муравьев (агентов).
        ranges (list[FuzzyBounds]): Границы для нечетких значений.
        r (np.ndarray): Входные данные для расчета потерь.
        R (int): Количество повторений для расчета потерь.

    Attributes:
        n (int): Количество входных переменных.
        p (int): Количество наблюдений.
        N (int): Общее количество параметров.
        R (int): Количество повторений для расчета потерь.
        X (pd.DataFrame): Датафрейм, содержащий входные переменные.
        t (np.ndarray): Целевые значения.
        r (np.ndarray): Входные данные для расчета потерь.
        ranges (list[FuzzyBounds]): Границы для нечетких значений.
        _low (float): Минимальное значение границ.
        _high (float): Максимальное значение границ.
        k (int): Количество начальных решений.
        theta (np.ndarray): Структура параметров.
        storage (list[Archive]): Хранилище для архива потерь и параметров.
        eps (float): Параметр для управления изменениями.
        q (float): Параметр для нормализации потерь.
        n_iter (int): Количество итераций алгоритма.
        n_ant (int): Общее количество муравьев (агентов).

    Methods:
        __f(theta): Вычисляет значение функции на основе параметров.


        _root_mean_squared_error(theta): Вычисляет среднеквадратическую ошибку между целевыми значениями и предсказанными значениями.
        _calc_weights(index): Вычисляет вес на основе позиции муравья.
        __init_solution(): Инициализирует решения, рассчитывая потери для каждого из них.
        best_result: Возвращает лучшее решение на данный момент.
        continuous_ant_algorithm(): Запускает непрерывный алгоритм муравьиных колоний.
        interp(num: FuzzyNumber, value: Union[int, float]) -> float: Интерполяция нечеткого числа на заданном значении.
        __construct_fuzzy_num(theta: np.ndarray) -> np.ndarray: Конструирует нечеткие числа из параметров.

    """

    def __init__(self, data: np.ndarray,
                 k: int, epsilon: float,
                 q: float, n_iter: int, n_ant: int, ranges: list[FuzzyBounds],
                 r: np.ndarray, n_terms: int, mf_type: str):

        self.n = data.shape[1] - 1
        self.p = data.shape[0]
        self.mf_type = mf_type
        self.n_terms = n_terms  # Число термов на каждый х
        self.__params = params[self.mf_type]
        self.N = self.__params * self.n_terms * self.n
        # self.R = R
        self.R = r.shape[0]  # число правил в базе
        self.X = pd.DataFrame(data={"x_" + str(i + 1): data[:, i] for i in range(self.n)})
        print(self.X)
        self.t = data[:, -1]
        self.r = r
        self.ranges = ranges
        self._low = min([f_bound.start for f_bound in self.ranges])
        self._high = max([f_bound.end for f_bound in self.ranges])

        self.k = k
        self.theta = np.reshape(np.random.uniform(low=self._low,
                                                  high=self._high,
                                                  size=(self.k * self.X.shape[0], self.N)),
                                (self.k, self.p, self.n * self.n_terms,
                                 self.__params))
        self.storage = [
            Archive(k=i, params=self.theta[i, :, :, :], loss=0) for i in range(self.k)
        ]

        self.theta.sort()
        self.eps = epsilon
        self.q = q
        self.n_iter = n_iter
        self.n_ant = n_ant

    def __f(self, theta: np.ndarray) -> float:
        """
        Вычисляет значение функции на основе параметров.

        Args:
            theta (np.ndarray): Параметры для вычисления функции.

        Returns:
            float: Вычисленное значение функции.
        """

        mu_value = self.__construct_fuzzy_num(theta)
        prod_value = np.prod(mu_value, axis=-1)
        noise = np.random.rand(*prod_value.shape)
        prod_value += noise
        _f = np.sum(prod_value * self.r, axis=-1) / np.sum(prod_value, axis=-1)
        return _f

    def _root_mean_squared_error(self, theta: np.ndarray) -> float:
        """
        Вычисляет среднеквадратическую ошибку между целевыми значениями и предсказанными значениями.

        Args:
            theta (np.ndarray): Параметры для вычислений.

        Returns:
            float: Среднеквадратическая ошибка.
        """

        _f = self.__f(theta)
        return np.sqrt(
            np.sum(
                np.square(self.t - _f)
            )
        ) / self.R

    def _calc_weights(self, index: int) -> float:
        """
        Вычисляет вес на основе позиции муравья.

        Args:
            index (int): Индекс муравья.

        Returns:
            float: Вычисленный вес.
        """

        return np.exp(-np.square(index - 1) / (2 * np.square(self.q) * np.square(self.k))) \
               / (self.q * self.k * np.sqrt(np.pi * 2))

    def __init_solution(self) -> None:
        """
        Инициализирует решения, рассчитывая потери для каждого из них.
        """

        for i in range(self.k):
            loss = self._root_mean_squared_error(self.storage[i].params)
            self.storage[i].loss = loss

        self.storage.sort(key=lambda x: x.loss)
        return

    @property
    def best_result(self) -> Archive:
        """
        Возвращает лучшее решение на данный момент.

        Returns:
            Archive: Лучшее решение с наименьшими потерями.
        """
        self.storage.sort(key=lambda x: x.loss)
        return self.storage[0]

    def continuous_ant_algorithm(self) -> np.ndarray:
        """
        Запускает непрерывный алгоритм муравьиных колоний.

        Returns:
            np.ndarray: Итоговые параметры после выполнения оптимизации.
        """

        self.__init_solution()

        n_fun = self.storage[0].params.shape[1]
        ant_per_groups = self.n_ant // n_fun

        for i in range(self.n_iter):

            for ant in range(ant_per_groups):

                theta = np.array([archive.params for archive in self.storage])
                sigma = np.zeros((self.p, n_fun, self.__params))

                for j in range(self.k - 1):
                    sub = np.abs(theta[0, :, :, :] - theta[j + 1, :, :, :])
                    sigma += sub

                sigma *= (self.eps / (self.k - 1))

                for j in range(self.k):
                    new_theta = vector_gaussian_f(theta[j, :, :, :], sigma)
                    new_theta = np.sort(new_theta)

                    new_loss = self._root_mean_squared_error(new_theta)

                    old_loss = self.storage[j].loss

                    if new_loss < old_loss:
                        self.storage[j].params = new_theta
                        self.storage[j].loss = new_loss

        self.storage.sort(key=lambda x: x.loss)
        th = self.storage[0].params

        return th

    @staticmethod
    def interp(num: FuzzyNumber, value: Union[int, float]):
        x, y = num.domain.x, num.values
        y_inter = interpolate.interp1d(x, y)
        return y_inter(value)

    def __construct_fuzzy_num(self,
                              theta: np.ndarray) -> np.ndarray:
        X_f_nums = np.zeros((self.p, self.R, self.n))
        # print(theta)
        # print(theta.shape)
        # X_f_nums = np.zeros((self.p, self.n, self.R))
        for line in self.ranges:
            x_name = line.x
            # for x_name in x_names:
            dom = Domain((line.start, line.end, line.step), name=x_name)
            _, i = tuple(x_name.split("_"))
            ind = int(i) - 1
            f_nums = np.array([
                [
                    dom.create_number(self.mf_type, *theta[l, ind + j, :].tolist()) for j in range(0,
                                                                                                   theta.shape[1],
                                                                                                   self.n)
                ] for l in range(theta.shape[0])
            ])
            # mu_arr = np.array([
            #     [
            #         self.interp(f_nums[i, j], self.X[x_name][i]) for j in range(f_nums.shape[1])
            #     ] for i in range(f_nums.shape[0])
            # ])
            # print(f_nums)
            mu_arr = np.array([
                [
                    f_nums[i, j](self.X[x_name][i]).item() for j in range(f_nums.shape[1])
                ] for i in range(f_nums.shape[0])
            ])
            # print(mu_arr)
            # print(X_f_nums)
            X_f_nums[:, :, ind] = mu_arr
        # print(X_f_nums)
        return X_f_nums

    # def __construct_fuzzy_num(self,
    #                           theta: np.ndarray) -> np.ndarray:
    #     X_f_nums = np.zeros((self.p, self.R, self.n))
    #     for line in self.ranges:
    #         x_names = line.x
    #         for x_name in x_names:
    #             dom = Domain((line.start, line.end, line.step), name=x_name)
    #             _, i = tuple(x_name.split("_"))
    #             ind = int(i) - 1
    #             f_nums = np.array([
    #                 [
    #                     dom.create_number(self.mf_type, *theta[l, ind + j, :].tolist()) for j in range(0,
    #                                                                                                    theta.shape[1],
    #                                                                                                    self.n)
    #                 ] for l in range(theta.shape[0])
    #             ])
    #             print(f_nums)
    #             # mu_arr = np.array([
    #             #     [
    #             #         self.interp(f_nums[i, j], self.X[x_name][i]) for j in range(f_nums.shape[1])
    #             #     ] for i in range(f_nums.shape[0])
    #             # ])
    #             mu_arr = np.array([
    #                 [
    #                     f_nums[i, j](self.X[x_name][i]).item() for j in range(f_nums.shape[1])
    #                 ] for i in range(f_nums.shape[0])
    #             ])
    #             print(mu_arr, mu_arr.shape)
    #             print(X_f_nums, X_f_nums.shape)
    #             X_f_nums[:, :, ind] = mu_arr
    #     return X_f_nums
