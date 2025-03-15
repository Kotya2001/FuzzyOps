from .neuron import FuzzyNNNeuron
from .synapse import FuzzyNNSynapse
from fuzzyops.fuzzy_numbers import Domain


class FuzzyNNLayer:
    """
    Представляет слой нечеткой нейронной сети, состоящий из нескольких нечетких нейронов.

    Attributes:
        _index (int): Индекс слоя в сети.
        _neurons (List[FuzzyNNNeuron]): Список нечетких нейронов в слое.
        _domain (Domain): Объект домена для работы с нечеткими числами.

    Args:
        ind (int): Индекс слоя.
        size (int): Количество нейронов в слое.
        domain (Domain): Объект домена для работы с нечеткими числами.
        neuronType (str): Тип нейронов в слое.

    Methods:
        add_into_synapse(toAddNeuronNumber: int, Synapse: FuzzyNNSynapse) -> None:
            Добавляет входящее ребро к указанному нейрону в слое.

        add_out_synapse(toAddNeuronNumber: int, Synapse: FuzzyNNSynapse) -> None:
            Добавляет исходящее ребро от указанного нейрона в слое.

        __len__() -> int:
            Возвращает количество нейронов в слое.

        forward() -> None:
            Проводит прямое распространение сигналов через все нейроны в слое.

        backward() -> None:
            Проводит обратное распространение ошибок через все нейроны в слое.
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
        Добавляет входящее ребро к указанному нейрону в слае.

        Args:
            toAddNeuronNumber (int): Индекс нейрона, к которому добавляется входящее ребро.
            Synapse (FuzzyNNSynapse): Синапс, который добавляется как входящее ребро.
        """

        self._neurons[toAddNeuronNumber].addInto(Synapse)

    def add_out_synapse(self, toAddNeuronNumber: int, Synapse: FuzzyNNSynapse) -> None:
        """
        Добавляет исходящее ребро от указанного нейрона в слое.

        Args:
            toAddNeuronNumber (int): Индекс нейрона, от которого добавляется исходящее ребро.
            Synapse (FuzzyNNSynapse): Синапс, который добавляется как исходящее ребро.
        """

        self._neurons[toAddNeuronNumber].addOut(Synapse)

    def __len__(self) -> int:
        """
        Возвращает количество нейронов в слое.

        Returns:
            int: Количество нейронов.
        """

        return len(self._neurons)

    def forward(self) -> None:
        """
        Проводит прямое распространение сигналов через все нейроны в слое.
        """

        for neuron in self._neurons:
            neuron.forward()

    def backward(self) -> None:
        """
        Проводит обратное распространение ошибок через все нейроны в слое.
        """

        for neuron in self._neurons:
            neuron.backward()
