from .neuron import FuzzyNNNeuron
from .synapse import FuzzyNNSynapse
from fuzzyops.fuzzy_numbers import Domain


class FuzzyNNLayer:
    """
    It represents a fuzzy neural network layer consisting of several fuzzy neurons

    Attributes:
        _index (int): The index of the layer in the network
        _neurons (List[FuzzyNNNeuron]): List of fuzzy neurons in the layer
        _domain (Domain): Domain object for working with fuzzy numbers

    Args:
        ind (int): Layer index
        size (int): Number of neurons per layer
        domain (Domain): Domain object for working with fuzzy numbers
        neuronType (str): The type of neurons in the layer
    """
    def __init__(
            self,
            ind: int,
            size: int,
            domain: Domain,
            neuronType: str,
    ):
        self._index = ind
        self._neurons = []
        self._domain = domain
        for i in range(size):
            neuron = FuzzyNNNeuron(neuronType)
            self._neurons.append(neuron)

    def add_into_synapse(self, toAddNeuronNumber: int, Synapse: FuzzyNNSynapse) -> None:
        """
        Adds an incoming edge to the specified neuron in the layer

        Args:
            toAddNeuronNumber (int): The index of the neuron to which the incoming edge is added
            Synapse (FuzzyNNSynapse): A synapse that is added as an incoming edge
        """

        self._neurons[toAddNeuronNumber].addInto(Synapse)

    def add_out_synapse(self, toAddNeuronNumber: int, Synapse: FuzzyNNSynapse) -> None:
        """
        Adds an outgoing edge from the specified neuron in the layer

        Args:
            toAddNeuronNumber (int): The index of the neuron from which the outgoing edge is added
            Synapse (FuzzyNNSynapse): A synapse that is added as an outgoing edge
        """

        self._neurons[toAddNeuronNumber].addOut(Synapse)

    def __len__(self) -> int:
        """
        Returns the number of neurons in the layer

        Returns:
            int: Number of neurons
        """

        return len(self._neurons)

    def forward(self) -> None:
        """
        It forwards signals through all neurons in the layer.
        """

        for neuron in self._neurons:
            neuron.forward()

    def backward(self) -> None:
        """
        It performs backpropagation of errors through all neurons in the layer
        """

        for neuron in self._neurons:
            neuron.backward()
