from .synapse import FuzzyNNSynapse
from fuzzyops.fuzzy_numbers import FuzzyNumber
from typing import Union


class Linear:
    """
    Класс для линейной активационной функции.

    Methods:
        forward(x: Union[FuzzyNumber, float, int]) -> Union[FuzzyNumber, float, int]:
            Применяет линейную активационную функцию к входному значению.

        backward(x: Union[FuzzyNumber, float, int]) -> Union[FuzzyNumber, float, int]:
            Возвращает производную линейной функции.
    """

    @staticmethod
    def forward(x: Union[FuzzyNumber, float, int]):
        return x

    @staticmethod
    def backward(x: Union[FuzzyNumber, float, int]):
        return x


class Relu:
    """
    Класс для активационной функции ReLU (Rectified Linear Unit).

    Methods:
        forward(x: Union[FuzzyNumber, float, int]) -> Union[FuzzyNumber, float, int]:
            Применяет активационную функцию ReLU к входному значению.

        backward(x: Union[FuzzyNumber, float, int]) -> Union[FuzzyNumber, float, int]:
            Возвращает производную функции ReLU.
    """

    @staticmethod
    def forward(x: Union[FuzzyNumber, float, int]):
        return 0 if float(x) <= 0 else x

    @staticmethod
    def backward(x: Union[FuzzyNumber, float, int]):
        return x


class FuzzyNNNeuron:
    """
    Представляет нечеткий нейрон в нечеткой нейронной сети.

    Attributes:
        neuronType (str): Тип нейрона (например, 'linear', 'relu').
        intoSynapses (List[FuzzyNNSynapse]): Список входящих синапсов.
        outSynapses (List[FuzzyNNSynapse]): Список исходящих синапсов.

    Args:
        neuronType (str): Тип нейрона (по умолчанию 'linear').

    Methods:
        addInto(toAdd: FuzzyNNSynapse) -> None:
            Добавляет входящий синапс к нейрону.

        addOut(toAdd: FuzzyNNSynapse) -> None:
            Добавляет исходящий синапс от нейрона.

        doCalculateForward(value: Union[FuzzyNumber, float, int]) -> None:
            Выполняет вычисление прямого распространения для данного значения.

        doCalculateBackward(value: Union[FuzzyNumber, float, int]) -> None:
            Выполняет вычисление обратного распространения для данного значения.

        forward() -> None:
            Проводит прямое распространение через нейрон.

        backward() -> None:
            Проводит обратное распространение через нейрон.
    """

    def __init__(self, neuronType: str = "linear"):
        self.neuronType = neuronType
        self.intoSynapses = []
        self.outSynapses = []

    def addInto(self, toAdd: FuzzyNNSynapse) -> None:
        """
        Добавляет входящий синапс к нейрону.

        Args:
            toAdd (FuzzyNNSynapse): Синапс, который будет добавлен в входящие синапсы.
        """

        self.intoSynapses.append(toAdd)

    def addOut(self, toAdd: FuzzyNNSynapse) -> None:
        """
        Добавляет исходящий синапс от нейрона.

        Args:
            toAdd (FuzzyNNSynapse): Синапс, который будет добавлен в исходящие синапсы.
        """

        self.outSynapses.append(toAdd)

    def doCalculateForward(self, value: Union[FuzzyNumber, float, int]) -> None:
        """
        Выполняет вычисление прямого распространения для данного значения.

        Args:
            value (Union[FuzzyNumber, float, int]): Входное значение для активации.

        Returns:
            Union[FuzzyNumber, float, int]: Результат активационной функции.
        """
        if self.neuronType == "linear":
            return Linear.forward(value)
        elif self.neuronType == "relu":
            return Relu.forward(value)

    def doCalculateBackward(self, value: Union[FuzzyNumber, float, int]) -> None:
        """
        Выполняет вычисление обратного распространения для данного значения.

        Args:
            value (Union[FuzzyNumber, float, int]): Ошибка для активации.

        Returns:
            Union[FuzzyNumber, float, int]: Производная активационной функции.
        """

        if self.neuronType == "linear":
            return Linear.backward(value)
        elif self.neuronType == "relu":
            return Relu.backward(value)

    def forward(self) -> None:
        """
        Проводит прямое распространение через нейрон.
        Накапливает значения входящих синапсов и передает результат
        через выходящие синапсы.
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
        Проводит обратное распространение через нейрон.
        Накапливает ошибки выходящих синапсов и передает результат
        через входящие синапсы.
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
