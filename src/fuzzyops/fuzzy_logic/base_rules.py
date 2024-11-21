from fuzzyops.fuzzy_numbers import Domain

from dataclasses import dataclass
from typing import Union


@dataclass
class BaseRule:
    """
    Базовое правило для нечеткой логики, содержащее нечеткое множество и значение.

    Attributes:
        A (Domain): Нечеткое множество, которое используется в правиле.
        B (Union[float, int, str]): Значение, ассоциированное с правилом.
    """
    A: Domain
    B: Union[float, int, str]


class FuzzyInference:
    """
    Класс для нечеткого вывода, использующий набор правил для формирования вывода.

    Attributes:
        domain (Domain): Нечеткая область, используемая для дискретизации выходных данных.
        ruleset (list[BaseRule]): Набор правил, применяемых в процессе вывода.
        defuzz_by (str): Метод дефаззификации, используемый для получения четкого значения. По умолчанию 'cgrav'.

    Methods:
        __call__(X: Union[float, int]) -> float: Применяет нечеткую
         логику к входному значению X и возвращает четкое значение.
    """

    def __init__(self,
                 domain: Domain,
                 ruleset: list[BaseRule],
                 defuzz_by: str = "cgrav"):
        self.domain = domain
        self.ruleset = ruleset
        self.defuzz_by = defuzz_by

    def __call__(self, X: Union[float, int]) -> float:
        """
        Применяет нечеткую логику к входному значению X и возвращает четкое значение.

        Args:
            X (Union[float, int]): Входное значение, на основе которого будет производиться нечеткий вывод.

        Returns:
            float: Четкое значение, полученное в результате нечеткого вывода.
        """

        out = self.domain.get(self.ruleset[0].B).clip_upper(self.ruleset[0].A(X))
        for rule in self.ruleset[1:]:
            out += self.domain.get(rule.B).clip_upper(rule.A(X))
        return out.defuzz(self.defuzz_by)
