from typing import Union, Tuple, List

from fuzzyops.fuzzy_numbers import Domain, FuzzyNumber
from .layer import FuzzyNNLayer
from .synapse import FuzzyNNSynapse

_initial_values = {
    'triangular': (-1, 0, 1),
    'trapezoidal': (-1, 0, 1, 2),
    'gauss': (0, 1)
}


class FuzzyNNetwork:
    """
    Класс для создания нечеткой нейронной сети.

    Attributes:
        _layers (List[FuzzyNNLayer]): Список слоев нечеткой нейронной сети.
        _verbose (Callable): Функция для вывода отладочной информации.
        _errors (List[float]): Список ошибок на каждой эпохе.
        _total_err (float): Общая ошибка сети.
        _domain (Domain): Объект домена для работы с нечеткими числами.
        _input_synapses (List[FuzzyNNSynapse]): Список входных синапсов.
        _output_synapses (List[FuzzyNNSynapse]): Список выходных синапсов.

    Methods:
        fit(x_train: List[List[FuzzyNumber]], y_train: List[List[FuzzyNumber]], steps: int = 1) -> None:
            Обучает нечеткую нейронную сеть на заданных тренировочных данных.

        predict(x_predict: List[FuzzyNumber]) -> List[float]:
            Делает предсказание на основе входных данных.
    Args:
        layersSizes (Union[tuple, list]): Размеры слоев сети.
        domainValues (Tuple, optional): Значения домена для нечетких чисел (по умолчанию (0, 100)).
        method (str, optional): Метод работы с нечеткими числами (по умолчанию 'minimax').
        fuzzyType (str, optional): Тип нечеткой числовой функции (по умолчанию 'triangular').
        activationType (str, optional): Тип активационной функции (по умолчанию 'linear').
        cuda (bool, optional): Использовать ли GPU (по умолчанию False).
        verbose (bool, optional): Выводить ли отладочную информацию (по умолчанию False).
    """

    def __init__(
            self,
            layersSizes: Union[tuple, list],
            domainValues: Tuple = (0, 100),
            method: str = 'minimax',
            fuzzyType: str = "triangular",
            activationType: str = "linear",
            cuda: bool = False,
            verbose: bool = False,
    ):
        self._layers = []
        self._verbose = lambda step: None
        self._errors = []
        self._total_err = 0
        if verbose:
            self._verbose = lambda step: print(f"step: {step}")
        self._domain = Domain(domainValues, name='domain', method=method)
        if cuda:
            self._domain.to('cuda')
        for i in range(len(layersSizes)):
            layer = FuzzyNNLayer(i, layersSizes[i], self._domain, activationType)
            self._layers.append(layer)
        for i in range(2, len(layersSizes) - 1):
            for fromSize in range(len(self._layers[i - 1])):
                for toSize in range(len(self._layers[i])):
                    synapseWeight = self._domain.create_number(fuzzyType, *_initial_values[fuzzyType],
                                                               name=f'neuron{i - 1}_{fromSize}:{i}_{toSize}')
                    synapse = FuzzyNNSynapse(synapseWeight)
                    self._layers[i - 1].add_out_synapse(fromSize, synapse)
                    self._layers[i].add_into_synapse(toSize, synapse)

        self._input_synapses = []
        for i in range(len(self._layers[0])):
            synapse = FuzzyNNSynapse(1)
            self._layers[0].add_into_synapse(i, synapse)
            self._input_synapses.append(synapse)

        self._output_synapses = []
        for i in range(len(self._layers[-1])):
            synapse = FuzzyNNSynapse(1)
            self._layers[-1].add_into_synapse(i, synapse)
            self._output_synapses.append(synapse)

    def fit(self, x_train: List[List[FuzzyNumber]],
            y_train: List[List[FuzzyNumber]], steps: int = 1) -> None:
        """
        Обучает нечеткую нейронную сеть на заданных тренировочных данных.

        Args:
            x_train (List[List[FuzzyNumber]]): Тренировочные данные входных значений.
            y_train (List[List[FuzzyNumber]]): Целевые значения для тренировочных данных.
            steps (int, optional): Число эпох обучения (по умолчанию 1).

        Raises:
            AssertionError: Если размеры x_train и y_train отличаются, или если размеры x_train и y_train
            не соответствуют ожидаемым размерам входных и выходных синапсов соответственно.
        """

        for st in range(steps):
            if ((st % 10 == 0) or (steps < 30)) and (st != 0):
                self._verbose(st)
            assert len(x_train) == len(y_train), "X and y are different sizes"
            assert len(x_train[0]) == len(self._input_synapses), "Wrong size of X"
            assert len(y_train[0]) == len(self._output_synapses), "Wrong size of y"
            for idx in range(len(x_train)):
                x = x_train[idx]
                y = y_train[idx]
                for i in range(len(x)):
                    self._input_synapses[i].setValue(x[i])
                for layer in self._layers:
                    layer.forward()
                results = [synapse.getValue() for synapse in self._output_synapses]

                semi_err = 0

                for i in range(len(y)):
                    try:
                        error = (results[i] - y[i]) * (results[i] - y[i])
                    except Exception:
                        error = (y[i] - results[i]) * (y[i] - results[i])
                    semi_err += error
                    self._output_synapses[i].setError(error)
                self._errors.append((semi_err / len(y)).defuzz())
                for layer in self._layers:
                    layer.backward()
        self._total_err = sum(self._errors) / len(self._errors)

    def predict(self, x_predict: List[FuzzyNumber]) -> List[float]:
        """
        Делает предсказание на основе входных данных.

        Args:
            x_predict (List[FuzzyNumber]): Входные данные для предсказания.

        Returns:
            List[float]: Список значений предсказания.

        Raises:
            AssertionError: Если размер x_predict не соответствует количеству входных синапсов.
        """

        assert len(x_predict) == len(self._input_synapses), "Wrong size of X"
        for i in range(len(x_predict)):
            self._input_synapses[i].setValue(x_predict[i])
        for layer in self._layers:
            layer.forward()
        results = [synapse.getValue() for synapse in self._output_synapses]

        return results
