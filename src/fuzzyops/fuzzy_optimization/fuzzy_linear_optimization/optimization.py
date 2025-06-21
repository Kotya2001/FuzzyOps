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
    A class for solving multi-criteria linear optimization problems (on the CPU and on the GPU), where the following matrices are set:
        a matrix of coefficients before functions,
        a matrix of coefficients of constraints, and a
        vector of constraints

    Attributes:
        A (np.ndarray): A matrix of constraint coefficients
        b (np.ndarray): A vector of constraint values
        C (np.ndarray): The matrix of coefficients facing the functions
        task_type (str): The type of optimization problem (minimization - 'min', maximization - 'max')
        num_vars (int): The number of variables in the optimization problem
        num_crits (int): The number of criteria in the optimization problem
        num_cons (int): The number of constraints in the optimization problem

    Args:
        A (np.ndarray): The matrix of coefficients of constraints
        b (np.ndarray): Vector of constraint values
        C (np.ndarray): The matrix of coefficients facing the functions
        task_type (str): Type of optimization problem (minimization - 'min', maximization - 'max')
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
        A method for solving multicriteria linear optimization problems (on a CPU)

        Args:
            all_positive (bool): A flag for setting an additional restriction,
                that it is necessary to look for solutions among the positive values

        Returns:
            Tuple[float, np.ndarray]: Returns the value of the objective function and an array of optimal solutions

        Raises:
            ValueError: If the passed task type (task_type) is not 'min' or 'max'
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
        A method for solving multicriteria linear optimization problems (on GPU) using the pythorch optimizer

        Args:
            lr (float): The convergence step of the algorithm
            epochs (int): Number of iterations of the algorithm

        Returns:
            Tuple[float, np.ndarray]: Returns the value of the objective function and an array of optimal solutions

        Raises:
            ValueError: If the passed task type (task_type) is not 'min' or 'max'
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
    Checks whether the fuzzy number matches the LR type

    Args:
        number (FuzzyNumber): A fuzzy number to check

    Returns:
        bool: True if the fuzzy number is convex and unimodal, otherwise False
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
    Checks whether the transmitted matrix of fuzzy numbers matches the LR type

    Args:
        matrix (np.ndarray): A matrix of fuzzy numbers to check

    Returns:
        np.ndarray: True if all fuzzy numbers are convex and unimodal, otherwise False
    """
    return np.all(vectorized_check_LR_type(matrix))


def _calc_root_value(square_num: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates root values for a square fuzzy number

    Args:
        square_num (np.ndarray): Input array of fuzzy numbers to calculate the root

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays with roots
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
    Calculates a scalar value based on two-dimensional arrays

    Args:
        c1 (np.ndarray): The first array of element-wise coefficients
        c2 (np.ndarray): The second array of element-wise coefficients

    Returns:
        np.ndarray: Calculated result as an array of scalar values
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
    Calculates a matrix and a dictionary containing information about the number of functions
    They cooperate, conflict, and are independent of each other

    Args:
        table (np.ndarray): A matrix for accounting for the number of different types of interactions
        k (np.ndarray): The matrix of interaction coefficients
        total_info (Dict[str, List[List[int]]]): A dictionary where the keys are the types of interaction (Cooperation, conflict, independence), 
            and the values are a two-dimensional array (an array with values in the i-th position means,
            that the i-th objective function cooperates, conflicts, and is independent of specific (depending on the key)
            objective functions (ids of objective functions in the internal array))


    Returns:
        Tuple[np.ndarray, Dict[str, List[List[int]]]]: A matrix for accounting for the number of different types of interaction and
            a populated dictionary, where the keys are the types of interaction (Cooperation, conflict, independence),
            and the values are a two-dimensional array (an array with values in the i-th position means,
            that the i-th objective function cooperates, conflicts, and is independent of specific (depending on the key)
            objective functions (ids of objective functions in the internal array))
    """

    for index, _ in np.ndenumerate(k):
        row, col = index[0], index[1]

        if 0.5 <= k[row][col] <= 1:
            table[row][0] += 1
            total_info["сooperation"][row].append(col)
        elif -1 <= k[row][col] <= -0.5:
            table[row][1] += 1
            total_info["сonflict"][row].append(col)
        elif -0.5 < k[row][col] < 0.5:
            table[row][2] += 1
            total_info["independence"][row].append(col)

    return table, total_info


def get_interaction_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, Dict[str, List[List[int]]]]:
    """
    Creates coefficients of interaction between each objective function and a dictionary, where keys are types of interaction (Cooperation, conflict, independence),
        and values are a two-dimensional array (an array with values in the i-th position means,
        that the i-th objective function cooperates, conflicts, and is independent of specific (depending on the key)
        objective functions (ids of objective functions in the internal array)).

    The algorithm is implemented according to the article:

      Аристова Е.М. Алгоритм решения задачи нечеткой многоцелевой линейной оптимизации
      с помощью определения коэффициента взаимодействия между
      целевыми функциями // Вестник Воронежского государственного университета.
      Серия: Системный анализ и информационные технологии. 2017 № 2. С. 105-110.


    Args:
        matrix (np.ndarray): The input matrix of fuzzy numbers

    Returns:
        Tuple[np.ndarray, Dict[str, List[List[int]]]]: Returns a matrix of coefficients of interaction of objective functions
            and a filled dictionary, where the keys are the types of interaction (Cooperation, conflict, independence),
            and the values are a two-dimensional array (an array with values in the i-th position means,
            that the ith objective function cooperates, conflicts, and is independent of specific (depending on the key)
            objective functions (indexes of objective functions in the internal array))
    """
    n, m, _ = matrix.shape
    k, interactions = np.zeros((n, n)), np.zeros((n, 3))
    total_info = {"сooperation": [[] for _ in range(n)],
                  "сonflict": [[] for _ in range(n)],
                  "independence": [[] for _ in range(n)]}
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
    Calculates the coefficient value for a specific type of interaction between the target functions

    Args:
        with_ind (int): The index of the function to calculate the coefficient from
        indx (List[int]): The list of indexes of functions with which it is necessary to calculate the coefficient
        params (np.ndarray): A matrix of clear values (modal values)
            of coefficients before variables in objective functions

    Returns:
        np.ndarray: The final vector of coefficients in front of the variables of the generalized objective function
    """
    res = params[with_ind] * params[indx[0]]
    for i in range(1, len(indx)):
        res += (params[with_ind] + params[indx[i]])
    return res


def calc_total_functions(alphs: np.ndarray, params: np.ndarray,
                         interaction_info: Dict[str, List[List[int]]], n: int) -> np.ndarray:
    """
    Calculates the final clear values of coefficients in a multi-objective linear optimization problem with
    fuzzy coefficients, based on the interaction coefficients and which specific functions
    cooperate, conflict, and are independent of each other

    Args:
        alphs (np.ndarray): Matrix of interaction coefficients
        params (np.ndarray): A matrix of clear values (modal values)
            of coefficients before variables in objective functions
        interaction_info (Dict[str, List[List[int]]]): A dictionary where the keys are the types of interaction (Cooperation, conflict, independence),
            and the values are a two-dimensional array (an array with values in the ith position means,
            that the ith objective function cooperates, conflicts, and is independent of specific (depending on the key)
            objective functions (indexes of objective functions in the internal array))
        n: (int): Number of target functions


    Returns:
        np.ndarray: The final vector of coefficients in front of the variables of the generalized objective function
    """

    arrays = []
    m = params.shape[1]
    for i in range(n):
        coop_coef = alphs[i, 0]
        conflict_coef = alphs[i, 1]
        indep_coef = alphs[i, 2]

        if len(interaction_info["сooperation"][i]):
            coop_res = __calc(i, interaction_info["сooperation"][i], params)
        else:
            coop_res = np.zeros((1, m))

        if len(interaction_info["сonflict"][i]):
            conf_res = __calc(i, interaction_info["сonflict"][i], params)
        else:
            conf_res = np.zeros((1, m))

        if len(interaction_info["independence"][i]):
            indep_res = __calc(i, interaction_info["independence"][i], params)
        else:
            indep_res = np.zeros((1, m))

        res = coop_coef * coop_res \
              + conflict_coef * conf_res \
              + indep_coef * indep_res
        arrays.append(res)

    combined_array = np.vstack(arrays)
    return np.sum(combined_array, axis=0)
