from fuzzyops.fuzzy_numbers import Domain, FuzzyNumber

from typing import Union, List
import numpy as np
import pandas as pd
from dataclasses import dataclass

tps = Union[FuzzyNumber, float]

params = {"triangular": 3, "trapezoidal": 4, "gauss": 2}


def __gaussian_f(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Generates normal random numbers based on the specified mean and standard deviation

    Args:
        mu (np.ndarray): The average value for a normal distribution
        sigma (np.ndarray): Standard deviation for a normal distribution

    Returns:
        np.ndarray: Generated normal random numbers
    """

    return np.random.default_rng().normal(loc=mu, scale=sigma)


vector_gaussian_f = np.vectorize(__gaussian_f)


@dataclass(frozen=True)
class FuzzyBounds:
    """
    Defines boundaries for fuzzy values

    Attributes:
        start (Union[int, float]): The initial value of the boundaries
        end (Union[int, float]): The final value of the boundaries
        step (Union[int, float]): A step for boundaries
        x (list[str]): A list of placemarks or border names
    """

    start: Union[int, float]
    end: Union[int, float]
    step: Union[int, float]
    x: str


@dataclass
class Archive:
    """
    Stores parameters and results for one optimization iteration

    Attributes:
        k (int): Archive index
        params (np.ndarray): Parameters of the fuzzy model
        loss (float): Loss (error) for this iteration
    """

    k: int
    params: np.ndarray
    loss: float


class AntOptimization:
    """
    Ant colony optimization algorithm for fuzzy systems identification
    The algorithm is based on the article:
    И.А. Ходашинский, П.А. Дудин. Идентификация нечетких систем на основе непрерывного алгоритма муравьиных колоний.
    Автометрия. 2012. Т. 48, № 1.

    Args:
        data (np.ndarray): Input data for creating a model (matrix of objects and features with a target variable)
        k (int): The size of the solution archive
        epsilon (float): A parameter that has the effect of evaporating pheromone in the discrete version of the algorithm
        q (float): A parameter for loss normalization
        n_iter (int): The number of iterations of the algorithm
        n_ant (int): The total number of ants (agents)
        ranges (List[FuzzyBounds]): Boundaries for fuzzy values
        r (np.ndarray): Input data for calculating losses (the value of consequents is a target variable)
        n_terms (int): The number of terms (usually equal to the number of observations)
        mf_type (str): The type of membership function for the created fuzzy numbers (Only triangular numbers, 'triangular')
        base_rules_ind (np.ndarray): An array of indices from 0 to n_terms

    Attributes:
        n (int): The number of input variables
        p (int): The number of observations
        N (int): The total number of parameters
        R (int): Number of rules in the rule base
        X (pd.DataFrame): A data frame containing input variables (matrix objects features)
        t (np.ndarray): Target values (target variable in the matrix of objects and features)
        r (np.ndarray): Input data for calculating losses
        base_rules_ind (np.ndarray): An array of indices from 0 to n_terms
        ranges (List[FuzzyBounds]): Boundaries for fuzzy values
        k (int): Number of initial solutions
        theta (np.ndarray): The structure of the parameters
        storage (List[Archive]): Storage for loss archive and parameters
        eps (float): A parameter that has the effect of evaporating pheromone in the discrete version of the algorithm
        q (float): Parameter for loss normalization
        n_iter (int): Number of iterations of the algorithm
        n_ant (int): Total number of ants (agents)
        n_colony (int): Ant colony size

    Properties:
        best_result() -> Archive:
            Returns the best solution at the moment

    Raises:
        ValueError: If a different type of membership function is passed, it is not a triangular function

    """

    def __init__(self, data: np.ndarray,
                 k: int, epsilon: float,
                 q: float, n_iter: int, n_ant: int, ranges: list[FuzzyBounds],
                 r: np.ndarray, n_terms: int, mf_type: str, base_rules_ind: np.ndarray):

        self.n = data.shape[1] - 1  # число входных переменных
        self.p = data.shape[0]  # число наблюдений
        if mf_type != "triangular":
            raise ValueError("Only triangular numbers are possible")
        self.mf_type = mf_type
        self.n_terms = n_terms  # Число термов на каждый х
        self.__params = params[self.mf_type]
        self.N = self.__params * self.n_terms * self.n

        self.R = r.shape[0]  # число правил в базе

        self.X = pd.DataFrame(data={"x_" + str(i + 1): data[:, i] for i in range(self.n)})  # обучающие данные
        self.t = data[:, -1]  # y в выборке

        self.r = r  # консеквенты
        self.base_rules_ind = base_rules_ind

        self.ranges = ranges
        self.k = k

        self.theta = self.__generate_theta()

        self.theta.sort()
        self.storage = [
            Archive(k=i, params=self.theta[i, :, :, :], loss=0) for i in range(self.k)
        ]

        self.eps = epsilon
        self.q = q
        self.n_iter = n_iter
        self.n_ant = n_ant
        self.n_colony = self.n * self.n_terms

    def __generate_theta(self) -> np.ndarray:
        """
        Generates initial parameters for membership functions before optimization

        Returns:
            np.ndarray: Initial parameters for membership functions

        """
        theta = np.zeros((self.k, self.n, self.n_terms, self.__params))
        for j in range(self.n):
            rng = self.ranges[j]
            low = rng.start
            high = rng.end

            theta[:, j, :, :] = np.random.uniform(low=low, high=high, size=(self.k,
                                                                            self.n_terms,
                                                                            self.__params))
        return theta

    def __f(self, theta: np.ndarray) -> float:
        """
        Calculates the value of a function based on its parameters

        Args:
            theta (np.ndarray): Parameters for calculating the function

        Returns:
            float: Calculated value of the function
        """

        mu_value = self.__construct_fuzzy_num(theta)
        noise = np.random.uniform(0, 0.01, size=mu_value.shape)
        matrix_with_noise = np.where(mu_value == 0.0, noise, mu_value)
        prod_value = np.prod(matrix_with_noise, axis=-1)
        _f = np.sum(prod_value * self.r, axis=-1) / np.sum(prod_value, axis=-1)
        return _f

    def _root_mean_squared_error(self, theta: np.ndarray) -> float:
        """
        Calculates the mean squared error between the target values and the predicted values

        Args:
            theta (np.ndarray): Parameters for calculations

        Returns:
            float: Standard deviation
        """

        _f = self.__f(theta)
        return np.sqrt(
            np.sum(
                np.square(self.t - _f)
            )
        ) / self.p

    def _calc_weights(self, index: int) -> float:
        """
        Calculates the weight based on the ant's position

        Args:
            index (int): The ant index

        Returns:
            float: Calculated weight
        """

        return np.exp(-np.square(index - 1) / (2 * np.square(self.q) * np.square(self.k))) \
               / (self.q * self.k * np.sqrt(np.pi * 2))

    def __init_solution(self) -> None:
        """
        Initializes the solutions by calculating the losses for each of them

        """
        for i in range(self.k):
            loss = self._root_mean_squared_error(self.storage[i].params)
            self.storage[i].loss = loss

        self.storage.sort(key=lambda x: x.loss)
        return

    @property
    def best_result(self) -> Archive:
        """
        Returns the best solution at the moment

        Returns:
            Archive: The best solution with the least losses
        """
        self.storage.sort(key=lambda x: x.loss)
        return self.storage[0]

    def continuous_ant_algorithm(self) -> np.ndarray:
        """
        Starts a continuous ant colony algorithm

        Returns:
            np.ndarray: Final parameters after optimization
        """

        self.__init_solution()

        n_fun = self.storage[0].params.shape[1]
        ant_per_groups = self.n_ant // n_fun

        for i in range(self.n_iter):

            for ant in range(ant_per_groups):

                theta = np.array([archive.params for archive in self.storage])
                sigma = np.zeros((self.k, self.n, self.n_terms, self.__params))

                for j in range(self.k - 1):
                    sub = np.abs(theta[0, :, :, :] - theta[j + 1, :, :, :])
                    sigma[j, :, :, :] += sub

                sigma *= (self.eps / (self.k - 1))

                for j in range(self.k):
                    new_theta = vector_gaussian_f(theta[j, :, :, :], sigma[j, :, :, :])
                    new_theta = np.sort(new_theta)

                    new_loss = self._root_mean_squared_error(new_theta)

                    old_loss = self.storage[j].loss

                    if new_loss < old_loss:
                        self.storage[j].params = new_theta
                        self.storage[j].loss = new_loss

        self.storage.sort(key=lambda x: x.loss)
        th = self.storage[0].params

        return th

    def __construct_fuzzy_num(self, theta: np.ndarray) -> np.ndarray:
        """
        Creates an array of confidence degrees based on the constructed fuzzy numbers with the found
        membership function parameters

        Args:
            theta (np.ndarray): Multidimensional array of membership function parameters

        Returns:
            np.ndarray: Multidimensional array of confidence levels
        """
        X_f_nums = np.zeros((self.p, self.R, self.n))
        for p in range(self.p):
            for line in self.ranges:
                x_name = line.x
                dom = Domain((line.start, line.end, line.step), name=x_name)
                _, i = tuple(x_name.split("_"))
                ind = int(i) - 1

                mu_arr = np.array([
                    dom.create_number(self.mf_type, *theta[ind, int(j), :].tolist())(self.X[x_name][p]).item()
                    for j in self.base_rules_ind[:, ind]
                ])
                X_f_nums[p, :, ind] = mu_arr

        return X_f_nums
