from fuzzyops.fuzzy_numbers import FuzzyNumber
import numpy as np
from uncertainties import ufloat
from typing import Union, Tuple, List, Dict
import torch

import cvxpy as cp

NumberTypes = Union["triangular"]
arrayTypes = Union[np.ndarray, torch.tensor]


class LinearOptimization:
    """
    Класс для решения задач многокритериальной линейной оптимизации (на ЦПУ и на ГПУ), где заданы следующие матрицы:
        матрица коэффициентов перед функциями,
        матрица коэффициентов ограниченийб
        вектор ограничений

    Attributes:
        A (np.ndarray): матрица коэффициентов ограничений.
        b (np.ndarray): вектор значений ограничений.
        C (np.ndarray): матрица коэффициентов, стоящих перед функциями.
        task_type (str): тип задачи оптимизации (минимизация - 'min', максимизация - 'max').
        num_vars (int): число переменных в задаче оптимизации.
        num_crits (int): число критериев в задаче оптимизации.
        num_cons (int): число ограничений в задаче оптимизации.

    Args:
        A (np.ndarray): матрица коэффициентов ограничений.
        b (np.ndarray): вектор значений ограничений.
        C (np.ndarray): матрица коэффициентов, стоящих перед функциями.
        task_type (str): тип задачи оптимизации (минимизация - 'min', максимизация - 'max')

    Methods:
        solve_cpu(all_positive: bool) -> Tuple[float, np.ndarray]:
            Решает задачу оптмиизации на ЦПУ, возвращает значение целевой функции и массив оптимальных решений
    """

    def __init__(self, A: np.ndarray, b: np.ndarray, C: np.ndarray, task_type):
        self.A = A
        self.b = b
        self.C = C
        self.task_type = task_type
        self.num_vars, self.num_crits, self.num_cons = A.shape[1], C.shape[0], b.shape[0]
        self.weights = np.ones(C.shape[0])

    def solve_cpu(self, all_positive=True):
        """
        Метод для решения задач многокритериальной линейной оптимизации (на ЦПУ)

        Args:
            all_positive (bool): Флаг для установки дополнительного ограничения,
                что необходимо искать решения среди положительных значений.

        Returns:
            Tuple[float, np.ndarray]: возвращает значение целевой функции и массив оптимальных решений.

        Raises:
            ValueError: Если переданный тип задачи (task_type) не 'min' и не 'max'.
        """
        x = cp.Variable(self.num_vars)
        mus = [self.C[i] @ x for i in range(self.num_crits)]
        mus_stacked = cp.vstack(mus)
        objective_value = self.weights @ mus_stacked

        if self.task_type == "max":
            objective = cp.Maximize(objective_value)
        elif self.task_type == "min":
            objective = cp.Minimize(objective_value)
        else:
            raise ValueError

        constraints = [self.A @ x <= self.b]
        if all_positive:
            constraints.append(x >= 0)

        prob = cp.Problem(objective, constraints)
        result = prob.solve()

        return result, x.value

    def solve_gpu(self, lr=0.001, epochs=10000):
        """
        Метод для решения задач многокритериальной линейной оптимизации (на ГПУ) с помощью оптимизаторв pythorh

        Args:
            lr (float): Шаг схождения алгоритма
            epochs (int): Число итерация алгоритма

        Returns:
            Tuple[float, np.ndarray]: возвращает значение целевой функции и массив оптимальных решений.

        Raises:
            ValueError: Если переданный тип задачи (task_type) не 'min' и не 'max'.
        """
        x = torch.randn(self.num_vars, device='cuda', requires_grad=True)
        C = torch.tensor(self.C.tolist(), dtype=torch.float32, device='cuda')
        b = torch.tensor(self.b.tolist(), dtype=torch.float32, device='cuda')
        A = torch.tensor(self.A.tolist(), dtype=torch.float32, device='cuda')

        optimizer = torch.optim.Adam([x], lr=lr)

        for _ in range(epochs):
            optimizer.zero_grad()
            mus = [torch.matmul(C[i], x) for i in range(self.num_crits)]
            if self.task_type == "max":
                objective = torch.max(torch.stack(mus))
            elif self.task_type == "min":
                objective = torch.min(torch.stack(mus))
            else:
                raise ValueError

            constraints_violation = (torch.matmul(A, x) - b).clamp(min=0).sum()
            loss = -objective + constraints_violation

            loss.backward()
            optimizer.step()

        return objective.item(), x.detach().cpu().numpy()


# check LR type of all nums in matrix, must be convex and unimodal
def _check_LR_type(number: FuzzyNumber) -> bool:
    """
    Проверяет, соответствует ли нечеткое число LR-типу.

    Args:
        number (FuzzyNumber): Нечеткое число для проверки.

    Returns:
        bool: True, если нечеткое число является выпуклым и унимодальным, иначе False.
    """

    values = number.values.tolist()
    np_values = np.array(values)
    membership_type = number.domain.membership_type
    _mu = np.where(np_values == 1.0)[0]
    if membership_type == "triangular":
        return _mu.size == 1
    return False


vectorized_check_LR_type = np.vectorize(_check_LR_type)


def check_LR_type(matrix: np.ndarray) -> np.ndarray:
    """
    Проверяет, соответствует ли переданная матрица нечетких чисел LR-типу.

    Args:
        matrix (np.ndarray): матрица нечетких чисел для проверки.

    Returns:
        np.ndarray: True, если все нечеткие числа являются выпуклыми и унимодальными, иначе False.
    """
    return np.all(vectorized_check_LR_type(matrix))



def _calc_root_value(square_num: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет корневые значения для квадратного нечеткого числа.

    Args:
        square_num (np.ndarray): Входной массив нечеткого числа, чтобы вычислить корень.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Два массива с корнями.
    """

    z1 = np.array([square_num[0] ** 0.5,
                   ufloat(square_num[0] ** 0.5, (square_num[0] - square_num[1]) ** 0.5),
                   ufloat(-1 * (square_num[0] ** 0.5), (square_num[0] + square_num[2]) ** 0.5)])

    z2 = np.array([square_num[0] ** 0.5 * -1,
                   ufloat((square_num[0] ** 0.5) * -1, (square_num[0] - square_num[1]) ** 0.5),
                   ufloat(square_num[0] ** 0.5, (square_num[0] + square_num[2]) ** 0.5)])
    return z1, z2


def _calc_scalar_value(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """
    Вычисляет скалярное значение на основе двумерных массивов.

    Args:
        c1 (np.ndarray): Первый массив поэлементных коэффициентов.
        c2 (np.ndarray): Второй массив поэлементных коэффициентов.

    Returns:
        np.ndarray: Вычисленный результат как массив скалярных значений.
    """

    res = c1.copy()
    res[0], res[1], res[2] = \
        c1[0] * c2[0], c1[0] * c2[1] + c2[0] * c1[1] - c1[1] * c2[1], \
        c1[0] * c2[2] + c2[0] * c1[2] + c1[2] * c2[2]

    return res


def _define_interaction_type(table: np.ndarray,
                             k: np.ndarray,
                             total_info: Dict[str, List[List[int]]]) -> Tuple[np.ndarray, Dict[str, List[List[int]]]]:
    """
    Рассчитывает матрицу и словарь, содержащую информацию о том какое количестов функций
    Кооперируют, конфликтуют, независыми друг с другом.

    Args:
        table (np.ndarray): Матрица для учета количества различных типов взаимодействия.
        k (np.ndarray): Матрица коэффициентов взаимодействий
        total_info (Dict[str, List[List[int]]]): Словарь, где ключи - это типы взаимодействия (Кооперация, конфликт, незавиисмость),
            а значения - двумерный массив (массив со значениями стоящий на i-ой позиции означает,
            что i-ая целевая функция кооперирую, конфликтует, независима с конкретными (в зависимости от ключа)
            целевымыи функциями (идексы целевых функций во внутреннем массиве))


    Returns:
        Tuple[np.ndarray, Dict[str, List[List[int]]]]: Матрица для учета количества различных типов взаимодействия и
        заполненный словарь, где ключи - это типы взаимодействия (Кооперация, конфликт, незавиисмость),
            а значения - двумерный массив (массив со значениями стоящий на i-ой позиции означает,
            что i-ая целевая функция кооперирую, конфликтует, независима с конкретными (в зависимости от ключа)
            целевымыи функциями (идексы целевых функций во внутреннем массиве))
    """

    for index, _ in np.ndenumerate(k):
        row, col = index[0], index[1]

        if 0.5 <= k[row][col] <= 1:
            table[row][0] += 1
            total_info["Кооперация"][row].append(col)
        elif -1 <= k[row][col] <= -0.5:
            table[row][1] += 1
            total_info["Конфликт"][row].append(col)
        elif -0.5 < k[row][col] < 0.5:
            table[row][2] += 1
            total_info["Независимость"][row].append(col)

    return table, total_info


def get_interaction_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, Dict[str, List[List[int]]]]:
    """
    Создает коэффициенты взаимодействия между каждой целевой функцией и словарь, де ключи - это типы взаимодействия (Кооперация, конфликт, незавиисмость),
     а значения - двумерный массив (массив со значениями стоящий на i-ой позиции означает,
     что i-ая целевая функция кооперирую, конфликтует, независима с конкретными (в зависимости от ключа)
     целевымыи функциями (идексы целевых функций во внутреннем массиве)).

    Алгоритм реализован по статье:

      Аристова Е.М. Алгоритм решения задачи нечеткой многоцелевой линейной оптимизации
      с помощью определения коэффициента взаимодействия между
      целевыми функциями // Вестник Воронежского государственного университета.
      Серия: Системный анализ и информационные технологии. 2017 № 2. С. 105-110.


    Args:
        matrix (np.ndarray): Входная матрица нечетких чисел.

    Returns:
        Tuple[np.ndarray, Dict[str, List[List[int]]]]: Возвращает матрицу коэффициентов взаимодействия целевых функций
        и заполненный словарь, где ключи - это типы взаимодействия (Кооперация, конфликт, незавиисмость),
            а значения - двумерный массив (массив со значениями стоящий на i-ой позиции означает,
            что i-ая целевая функция кооперирую, конфликтует, независима с конкретными (в зависимости от ключа)
            целевымыи функциями (идексы целевых функций во внутреннем массиве))
    """
    n, m, _ = matrix.shape
    k, interactions = np.zeros((n, n)), np.zeros((n, 3))
    total_info = {"Кооперация": [[] for _ in range(n)],
                  "Конфликт": [[] for _ in range(n)],
                  "Независимость": [[] for _ in range(n)]}
    np.fill_diagonal(k, 1)
    repeats = {}

    for row in range(n):
        for col in range(n):
            if row != col:
                total = row + col
                if str(total) not in list(repeats.keys()):
                    numerator = np.sum(np.array([_calc_scalar_value(matrix[row][l], matrix[col][l])
                                                 for l in range(m)]), axis=0)
                    square_sum_i = np.sum(np.array([_calc_scalar_value(matrix[row][l], matrix[row][l])
                                                    for l in range(m)]), axis=0)
                    square_sum_j = np.sum(np.array([_calc_scalar_value(matrix[col][l], matrix[col][l])
                                                    for l in range(m)]), axis=0)

                    root_i_1, root_i_2 = _calc_root_value(square_sum_i)
                    root_j_1, root_j_2 = _calc_root_value(square_sum_j)

                    root1, root2 = _calc_scalar_value(root_i_1, root_j_1), _calc_scalar_value(root_i_2, root_j_2)
                    res = numerator[0] / root1[0]
                    k[row][col] = res
                    repeats.update({str(total): (row, col, res)})
                else:
                    row, col, res = repeats[str(total)]
                    k[col][row] = res

                    del repeats[str(total)]
                    continue

    interactions, interactions_list = _define_interaction_type(interactions, k, total_info)
    alphas = interactions / n

    return alphas, interactions_list


def __calc(with_ind: int, indx: List[int], params: np.ndarray) -> np.ndarray:
    """
    Рассчитывает значение коэффициента по конкретному типу зваимодействия между целевыми функциями

    Args:
        with_ind (int): индекс функции с которой необходимо рассчитать коэффициент.
        indx (List[int]): Список индексов функций с которыми необходимо рассчитать коэффициент
        params (np.ndarray): Матрица четких значений (модальные значения)
            коэффициентов перед переменными в целевых функциях.

    Returns:
        np.ndarray: Итоговый вектор коэффициентов перед переменными у обобщенной целевой функции
    """
    res = params[with_ind] * params[indx[0]]
    for i in range(1, len(indx)):
        res += (params[with_ind] + params[indx[i]])
    return res


def calc_total_functions(alphs: np.ndarray, params: np.ndarray,
                         interaction_info: Dict[str, List[List[int]]], n: int) -> np.ndarray:
    """
    Рассчитывает итоговые четкие значения коэффициентов в задаче многоцелевой линейной оптимизации с
    нечеткими коэффициентами, на основе коэффицентов взаимодействия и то какие конкретно функции
    кооперируют, конфликтуют, независимы друг с другом

    Args:
        alphs (np.ndarray): матрица коэффициентов взаимодействия.
        params (np.ndarray): Матрица четких значений (модальные значения)
            коэффициентов перед переменными в целевых функциях.
        interaction_info (Dict[str, List[List[int]]]): Словарь, где ключи - это типы взаимодействия (Кооперация, конфликт, незавиисмость),
            а значения - двумерный массив (массив со значениями стоящий на i-ой позиции означает,
            что i-ая целевая функция кооперирую, конфликтует, независима с конкретными (в зависимости от ключа)
            целевымыи функциями (идексы целевых функций во внутреннем массиве))
        n: (int): число целевых функций


    Returns:
        np.ndarray: Итоговый вектор коэффициентов перед переменными у обобщенной целевой функции
    """

    arrays = []
    m = params.shape[1]
    for i in range(n):
        coop_coef = alphs[i, 0]
        conflict_coef = alphs[i, 1]
        indep_coef = alphs[i, 2]

        if len(interaction_info["Кооперация"][i]):
            coop_res = __calc(i, interaction_info["Кооперация"][i], params)
        else:
            coop_res = np.zeros((1, m))

        if len(interaction_info["Конфликт"][i]):
            conf_res = __calc(i, interaction_info["Конфликт"][i], params)
        else:
            conf_res = np.zeros((1, m))

        if len(interaction_info["Независимость"][i]):
            indep_res = __calc(i, interaction_info["Независимость"][i], params)
        else:
            indep_res = np.zeros((1, m))

        res = coop_coef * coop_res \
              + conflict_coef * conf_res \
              + indep_coef * indep_res
        arrays.append(res)

    combined_array = np.vstack(arrays)
    return np.sum(combined_array, axis=0)
