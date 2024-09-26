from typing import Union

from ..fuzzy_numbers import Domain
from .layer import FuzzyNNLayer
from .synapse import FuzzyNNSynapse

_initial_values = {
    'triangular': (-1, 0, 1),
    'trapezoidal': (-1, 0, 1, 2),
    'gauss': (0, 1)
}

class FuzzyNNetwork:
    def __init__(
        self,
        layersSizes: Union[tuple, list],
        domainValues = (0, 100),
        method: str = 'minimax',
        fuzzyType = "triangular",
        activationType = "linear",
        cuda: bool = False,
        verbose: bool = False,
    ):
        self._layers = []
        self._verbose = lambda step: None
        if verbose:
            self._verbose = lambda step: print(f"step: {step}")
        self._domain = Domain(domainValues, name='domain', method=method)
        if cuda:
            self._domain.to('cpu')
        for i in range(len(layersSizes)):
            layer = FuzzyNNLayer(i, layersSizes[i], self._domain, activationType)
            self._layers.append(layer)
        for i in range(2, len(layersSizes)-1):
            for fromSize in range(len(self._layers[i-1])):
                for toSize in range(len(self._layers[i])):
                    synapseWeight = self._domain.create_number(fuzzyType, *_initial_values[fuzzyType], name=f'neuron{i-1}_{fromSize}:{i}_{toSize}')
                    synapse = FuzzyNNSynapse(synapseWeight)
                    self._layers[i-1].add_out_synapse(fromSize, synapse)
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

    def fit(self, x_train, y_train, steps=1):
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

                for i in range(len(y)):
                    try:
                        error = (results[i] - y[i]) * (results[i] - y[i])
                    except Exception:
                        error = (y[i] - results[i]) * (y[i] - results[i])
                    self._output_synapses[i].setError(error)
                for layer in self._layers:
                    layer.backward()

    def predict(self, x_predict):
        assert len(x_predict) == len(self._input_synapses), "Wrong size of X"
        for i in range(len(x_predict)):
            self._input_synapses[i].setValue(x_predict[i])
        for layer in self._layers:
            layer.forward()
        results = [synapse.getValue() for synapse in self._output_synapses]

        return results