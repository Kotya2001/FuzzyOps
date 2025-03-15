from fuzzyops.fuzzy_numbers import Domain, FuzzyNumber

from typing import Union, Dict, List, Tuple
from dataclasses import dataclass

import numpy as np


@dataclass
class BaseRule:
    """
    Базовый класс для представления правила в базе знаний системы нечеткой логики.

    Attributes:
        antecedents (List[Tuple[str]]): Антецеденты правила, представляющие собой условия.
        consequent (Any): Консеквент правила, представляющий собой заключение.
    """
    antecedents: List[Tuple[str]]
    consequent: Tuple[str]


class FuzzyInference:
    """
    Класс для осуществления нечеткого логического вывода по алгоритму Мамдани

    Attributes:
        domains (Dict[str, Domain]): Словарь доменов для нечетких чисел.
        rules (List[BaseRule]): Список правил в базе правил.

    Args:
        domains (Dict[str, Domain]): Словарь доменов для нечетких чисел.
        rules (List[BaseRule]): Список правил в базе правил.

    Methods:
        compute(input_data: Dict[str, Union[int, float, FuzzyNumber]]) -> Dict[str, float]:
            Вычисляет дефаззифицированные значения консеквентов правил

    Raises:
        AttributeError: Если переданное имя домена не присутствует в базе правил
    """
    def __init__(self, domains: Dict[str, Domain], rules: List[BaseRule]):
        self.domains = domains
        self.rules = rules

    def compute(self, input_data: Dict[str, Union[int, float, FuzzyNumber]]) -> Dict[str, float]:

        """
        Метод, который вычисляет значения консетквентов в базе правил по алгоритму Мамдани

        Args:
            input_data (Dict[str, Union[int, float, FuzzyNumber]):
            Словарь с названиями доменов из базы правил и значениями из универсального множества (входные данные),
            для которых необходимо найти значения консеквентов (дефаззифицированные значения выходной переменной)

        Returns:
            Dict[str, float]: Словарь, ключ - название консеквента, значение - дефаззифицированный резульат.

        Raises:
            AssertionError: Если membership не строка или не соответствует необходимому числу аргументов.
        """

        results = {rule.consequent[0]: 0 for rule in self.rules}
        for rule in self.rules:
            antecedents = rule.antecedents
            consequent = rule.consequent
            res = 1
            cons_domain = self.domains.get(consequent[0])
            cons_ters = cons_domain.get(consequent[1])
            for antecedent in antecedents:
                domain_name = antecedent[0]
                if domain_name not in input_data.keys():
                    raise AttributeError("Недостаточно данных")
                value = input_data.get(domain_name)
                if value is None:
                    if domain_name in results.keys():
                        domain = self.domains.get(domain_name)
                        term = domain.get(antecedent[1])
                        values = term.values
                        res *= sum([cons_ters.clip_upper(v) for v in values])

                else:
                    if isinstance(value, FuzzyNumber):
                        values = value.values
                        res *= sum([cons_ters.clip_upper(v) for v in values])

                    else:
                        domain = self.domains.get(domain_name)

                        term = domain.get(antecedent[1])
                        res *= cons_ters.clip_upper(term(value))
            r = results.get(consequent[0])
            r += res
            results[consequent[0]] = r

        return results


class SingletonInference:
    """
    Класс для осуществления нечеткого логического вывода по алгоритму Синглтон

    Attributes:
        domains (Dict[str, Domain]): Словарь доменов для нечетких чисел.
        rules (List[BaseRule]): Список правил в базе правил.

    Args:
        domains (Dict[str, Domain]): Словарь доменов для нечетких чисел.
        rules (List[BaseRule]): Список правил в базе правил.

    Methods:
        compute(input_data: Dict[str, Union[int, float, FuzzyNumber]]) -> Dict[str, float]:
            Вычисляет дефаззифицированное значение
    """
    def __init__(self, domains: Dict[str, Domain], rules: List[BaseRule]):
        self.domains = domains
        self.rules = rules

    def compute(self, input_data: Dict[str, Union[int, float, FuzzyNumber]]) -> float:
        """
        Метод, который вычисляет значения консетквентов в базе правил по алгоритму Синглтон

        Args:
            input_data (Dict[str, Union[int, float, FuzzyNumber]):
            Словарь с названиями доменов из базы правил и значениями из универсального множества (входные данные),
            для которых необходимо найти значения консеквентов (дефаззифицированные значения выходной переменной)

        Returns:
            Dict[str, float]: Числовое значение консеквента
        """

        sorted_keys = [k[0] for k in self.rules[0].antecedents]
        inp = np.array([input_data[key] for key in sorted_keys if key in input_data])

        r = np.array([rule.consequent for rule in self.rules])
        mu_arr = np.array(
            [
                [self.domains[rule.antecedents[i][0]].get(rule.antecedents[i][1])(inp[i]).item()
                 for i in range(len(rule.antecedents))]
                for rule in self.rules
            ]
        )
        prod_value = np.prod(mu_arr, axis=-1)
        return np.sum(prod_value * r, axis=-1) / np.sum(prod_value, axis=-1)




