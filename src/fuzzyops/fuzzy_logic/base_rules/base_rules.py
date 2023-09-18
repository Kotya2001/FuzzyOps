from src.fuzzyops.fuzzy_numbers import Domain

from dataclasses import dataclass
from typing import Union


@dataclass
class BaseRule:
    X: Union[float, int]
    A: Domain
    B: Union[float, int]
