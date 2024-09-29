from src.fuzzyops.fuzzy_numbers import Domain

from dataclasses import dataclass
from typing import Union


@dataclass
class BaseRule:
    A: Domain
    B: Union[float, int]

class FuzzyInference:
    def __init__(self, 
                 domain: Domain,
                 ruleset: list[BaseRule],
                 defuzz_by: str = "cgrav"):
        self.domain = domain
        self.ruleset = ruleset
        self.defuzz_by = defuzz_by

    def __call__(self, X: Union[float, int]) -> float:
        out = self.domain.get(self.ruleset[0].B).clip_upper(self.ruleset[0].A(X))
        for rule in self.ruleset[1:]:
            out += self.domain.get(rule.B).clip_upper(rule.A(X))
        return out.defuzz(self.defuzz_by)
        