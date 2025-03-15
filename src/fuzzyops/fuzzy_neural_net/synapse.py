from typing import Union
from fuzzyops.fuzzy_numbers import FuzzyNumber


class FuzzyNNSynapse:
    """
    Представляет синапс в нечеткой нейронной сети, который соединяет два нейрона.

    Attributes:
        value (Union[float, FuzzyNumber]): Значение синапса, передаваемое через него.
        error (Union[float, FuzzyNumber]): Ошибка, связанная с синапсом, для обновления веса.
        weight (Union[float, FuzzyNumber]): Вес синапса, влияющий на передаваемое значение.

    Args:
        weight (Union[float, FuzzyNumber]): Начальный вес синапса.
    """

    def __init__(self, weight: Union[float, FuzzyNumber]):
        self.value = 0
        self.error = 0
        self.weight = weight

    def setValue(self, value: Union[float, FuzzyNumber]):
        """
        Устанавливает значение синапса.

        Args:
            value (Union[float, FuzzyNumber]): Значение, которое будет установлено для синапса.
        """

        self.value = value

    def getValue(self) -> FuzzyNumber:
        """
        Возвращает значение синапса.

        Returns:
            FuzzyNumber: Значение синапса, умноженное на его вес.
        """

        return self.value * self.weight

    def setError(self, error: Union[float, FuzzyNumber]):
        """
        Устанавливает ошибку синапса.

        Args:
            error (Union[float, FuzzyNumber]): Ошибка, которая будет установлена для синапса.
        """

        self.error = error

    def getError(self) -> None:
        """
        Возвращает текущую ошибку синапса.

        Returns:
            Union[float, FuzzyNumber]: Ошибка синапса.
        """

        return self.error

    def applyError(self) -> None:
        """
        Применяет ошибку к весу синапса, обновляя его значение.

        Обновление веса происходит по формуле:
        новый вес = старый вес + ошибка * 0.1
        """

        self.weight = self.weight + self.error * 0.1
