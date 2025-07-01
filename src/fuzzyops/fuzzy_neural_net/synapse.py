from typing import Union
from fuzzyops.fuzzy_numbers import FuzzyNumber


class FuzzyNNSynapse:
    """
    It represents a synapse in a fuzzy neural network that connects two neurons.

    Attributes:
        value (Union[float, FuzzyNumber]): The value of the synapse transmitted through it
        error (Union[float, FuzzyNumber]): A synapse-related error for weight updates
        weight (Union[float, FuzzyNumber]): The weight of the synapse that affects the transmitted value

    Args:
        weight (Union[float, FuzzyNumber]): The initial weight of the synapse
    """

    def __init__(self, weight: Union[float, FuzzyNumber]):
        self.value = 0
        self.error = 0
        self.weight = weight

    def setValue(self, value: Union[float, FuzzyNumber]):
        """
        Sets the synapse value

        Args:
            value (Union[float, FuzzyNumber]): The value that will be set for the synapse
        """

        self.value = value

    def getValue(self) -> FuzzyNumber:
        """
        Returns the synapse value

        Returns:
            FuzzyNumber: The value of a synapse multiplied by its weight
        """

        return self.value * self.weight

    def setError(self, error: Union[float, FuzzyNumber]):
        """
        Sets the synapse error

        Args:
            error (Union[float, FuzzyNumber]): An error that will be set for the synapse
        """

        self.error = error

    def getError(self) -> None:
        """
        Returns the current synapse error

        Returns:
            Union[float, FuzzyNumber]: Synapse error
        """

        return self.error

    def applyError(self) -> None:
        """
        Applies an error to the synapse weight, updating its value

        The weight is updated using the following formula:
        new weight = old weight + error * 0.1
        """

        self.weight = self.weight + self.error * 0.1
