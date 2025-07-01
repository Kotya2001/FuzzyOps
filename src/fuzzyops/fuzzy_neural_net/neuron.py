from .synapse import FuzzyNNSynapse
from fuzzyops.fuzzy_numbers import FuzzyNumber
from typing import Union


class Linear:
    """
    A class for a linear activation function
    """

    @staticmethod
    def forward(x: Union[FuzzyNumber, float, int]):
        return x

    @staticmethod
    def backward(x: Union[FuzzyNumber, float, int]):
        return x


class Relu:
    """
    Class for the ReLU (Rectified Linear Unit) activation function
    """

    @staticmethod
    def forward(x: Union[FuzzyNumber, float, int]):
        return 0 if float(x) <= 0 else x

    @staticmethod
    def backward(x: Union[FuzzyNumber, float, int]):
        return x


class FuzzyNNNeuron:
    """
    Represents a fuzzy neuron in a fuzzy neural network

    Attributes:
        neuronType (str): The type of neuron (for example, 'linear', 'relu')
        intoSynapses (List[FuzzyNNSynapse]): List of incoming synapses
        outSynapses (List[FuzzyNNSynapse]): List of outgoing synapses

    Args:
        neuronType (str): Type of neuron (default: 'linear')
    """

    def __init__(self, neuronType: str = "linear"):
        self.neuronType = neuronType
        self.intoSynapses = []
        self.outSynapses = []

    def addInto(self, toAdd: FuzzyNNSynapse) -> None:
        """
        Adds an incoming synapse to a neuron

        Args:
            toAdd (FuzzyNNSynapse): A synapse that will be added to the incoming synapses
        """

        self.intoSynapses.append(toAdd)

    def addOut(self, toAdd: FuzzyNNSynapse) -> None:
        """
        Adds an outgoing synapse from a neuron

        Args:
            toAdd (FuzzyNNSynapse): A synapse that will be added to the outgoing synapses
        """

        self.outSynapses.append(toAdd)

    def doCalculateForward(self, value: Union[FuzzyNumber, float, int]) -> None:
        """
        Calculates the forward propagation for a given value

        Args:
            value (Union[FuzzyNumber, float, int]): The input value for activation

        Returns:
            Union[FuzzyNumber, float, int]: The result of the activation function
        """
        if self.neuronType == "linear":
            return Linear.forward(value)
        elif self.neuronType == "relu":
            return Relu.forward(value)

    def doCalculateBackward(self, value: Union[FuzzyNumber, float, int]) -> None:
        """
        Calculates the backward propagation for a given value

        Args:
            value (Union[FuzzyNumber, float, int]): Error for activation

        Returns:
            Union[FuzzyNumber, float, int]: The derivative of the activation function.
        """

        if self.neuronType == "linear":
            return Linear.backward(value)
        elif self.neuronType == "relu":
            return Relu.backward(value)

    def forward(self) -> None:
        """
        Performs direct propagation through the neuron
        Accumulates the values of the incoming synapses and passes the result through the outgoing synapses.
        """

        z = 0
        for syn in self.intoSynapses:
            v = syn.getValue()
            try:
                z += v
            except Exception:
                z = v + z
        z = self.doCalculateForward(z)
        for syn in self.outSynapses:
            syn.setValue(z)

    def backward(self) -> None:
        """
        Performs backpropagation through the neuron
        Accumulates errors of outgoing synapses and passes the result through incoming synapses.
        """

        z = 0
        for syn in self.outSynapses:
            v = syn.getError()
            try:
                z += v
            except Exception:
                z = v + z
        z = self.doCalculateBackward(z)
        for syn in self.intoSynapses:
            syn.setError(z)
        for syn in self.outSynapses:
            syn.applyError()
