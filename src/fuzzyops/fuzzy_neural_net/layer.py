from .neuron import FuzzyNNNeuron


class FuzzyNNLayer:
    def __init__(
            self,
            ind,
            size,
            domain,
            neuronType,
    ):
        self._index = ind
        self._neurons = []
        self._domain = domain
        for i in range(size):
            neuron = FuzzyNNNeuron(neuronType)
            self._neurons.append(neuron)

    def add_into_synapse(self, toAddNeuronNumber, Synapse):
        self._neurons[toAddNeuronNumber].addInto(Synapse)

    def add_out_synapse(self, toAddNeuronNumber, Synapse):
        self._neurons[toAddNeuronNumber].addOut(Synapse)

    def __len__(self):
        return len(self._neurons)

    def forward(self):
        for neuron in self._neurons:
            neuron.forward()

    def backward(self):
        for neuron in self._neurons:
            neuron.backward()
