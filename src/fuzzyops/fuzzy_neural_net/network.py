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
    A class for creating a fuzzy neural network

    Attributes:
        _layers (List[FuzzyNNLayer]): List of layers in a fuzzy neural network.
        _verbose (Callable): A function for displaying debugging information
        _errors (List[float]): A list of errors for each epoch
        _total_err (float): General Network error
        _domain (Domain): Domain object for working with fuzzy numbers
        _input_synapses (List[FuzzyNNSynapse]): List of input synapses
        _output_synapses (List[FuzzyNNSynapse]): List of output synapses
    Args:
        layersSizes (Union[tuple, list]): Network layer sizes
        domainValues (Tuple, optional): Domain values for fuzzy numbers (default (0, 100))
        method (str, optional): Method for working with fuzzy numbers (default: 'minimax')
        fuzzyType (str, optional): Type of fuzzy numerical function (default is 'triangular')
        activationType (str, optional): Activation function type (default: 'linear')
        cuda (bool, optional): Use GPU (default False)
        verbose (bool, optional): Whether to display debug information (False by default)
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
        Trains a fuzzy neural network using the given training data

        Args:
            x_train (List[List[FuzzyNumber]]): Training data of input values
            y_train (List[List[FuzzyNumber]]): Target values for training data
            steps (int, optional): Number of training epochs (default: 1)

        Raises:
            AssertionError: If the sizes of x_train and y_train are different, or if the sizes of x_train and y_train
            do not match the expected sizes of the input and output synapses, respectively
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
        Makes a prediction based on the input data

        Args:
            x_predict (List[FuzzyNumber]): Input data for prediction

        Returns:
            List[float]: List of prediction values

        Raises:
            AssertionError: If the size of x_predict does not match the number of input synapses
        """

        assert len(x_predict) == len(self._input_synapses), "Wrong size of X"
        for i in range(len(x_predict)):
            self._input_synapses[i].setValue(x_predict[i])
        for layer in self._layers:
            layer.forward()
        results = [synapse.getValue() for synapse in self._output_synapses]

        return results
