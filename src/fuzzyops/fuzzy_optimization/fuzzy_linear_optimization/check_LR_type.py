"""
Проверка числе на LR-тип

Необходимо проверить нечеткое число на нормальность:

 - существует значение носителя, в котором функция принадлежности равна единице (условие нормальности);
 - при отступлении от своего максимума влево или вправо функция принадлежности не возрастает (условие выпуклости);

"""
from src.fuzzyops.fuzzy_numbers import FuzzyNumber
import numpy as np


def check_LR_type(fuzzy_number: FuzzyNumber) -> bool:
    """
    Check LR-type for fuzzy number

    :param fuzzy_number:
    :return: bool
    """
    values = fuzzy_number.values.numpy()
    membership_type = fuzzy_number.domain.membership_type
    _mu = np.where(values == 1.0)[0]
    if membership_type in ("triangular", "gauss"):
        return _mu.size == 1
    else:
        return _mu.size == 2
