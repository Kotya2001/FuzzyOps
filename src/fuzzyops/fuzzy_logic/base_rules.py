from fuzzyops.fuzzy_numbers import Domain, FuzzyNumber

from typing import Union, Dict, List, Tuple
from dataclasses import dataclass

import numpy as np


@dataclass
class BaseRule:
    """
    A base class for representing a rule in the knowledge base of a fuzzy logic system

    Attributes:
        antecedents (List[Tuple[str]]): Antecedents of the rule, representing the conditions
        consequent (Any): The rule's consequence, which is the conclusion
    """
    antecedents: List[Tuple[str]]
    consequent: Tuple[str]


class FuzzyInference:
    """
    A class for implementing fuzzy logical inference using the Mamdani algorithm

    Attributes:
        domains (Dict[str, Domain]): Dictionary of domains for fuzzy numbers
        rules (List[BaseRule]): List of rules in the rule database

    Args:
        domains (Dict[str, Domain]): Dictionary of domains for fuzzy numbers
        rules (List[BaseRule]): List of rules in the rule database

    Raises:
        AttributeError: If the transmitted domain name is not present in the rule database
    """
    def __init__(self, domains: Dict[str, Domain], rules: List[BaseRule]):
        self.domains = domains
        self.rules = rules

    def compute(self, input_data: Dict[str, Union[int, float, FuzzyNumber]]) -> Dict[str, float]:

        """
        A method that calculates the values of the coefficients in the rule base using the Mamdani algorithm

        Args:
            input_data (Dict[str, Union[int, float, FuzzyNumber]):
            A dictionary with domain names from the rule base and values from the universal set (input data)
            for which it is necessary to find the values of the consequences (defuzzified values of the output variable)

        Returns:
            Dict[str, float]: Dictionary, the key is the name of the consequence, the value is the defazzified result

        Raises:
            AssertionError: If membership is not a string or does not match the required number of arguments
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
                    raise AttributeError("Insufficient data")
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
    A class for implementing fuzzy logical inference using the Singleton algorithm

    Attributes:
        domains (Dict[str, Domain]): A dictionary of domains for fuzzy numbers
        rules (List[BaseRule]): The list of rules in the rule database

    Args:
        domains (Dict[str, Domain]): A dictionary of domains for fuzzy numbers
        rules (List[BaseRule]): The list of rules in the rule database
    """
    def __init__(self, domains: Dict[str, Domain], rules: List[BaseRule]):
        self.domains = domains
        self.rules = rules

    def compute(self, input_data: Dict[str, Union[int, float, FuzzyNumber]]) -> float:
        """
        A method that calculates the values of the consetquents in the rule base using the Singleton algorithm

        Args:
            input_data (Dict[str, Union[int, float, FuzzyNumber]):
            A dictionary with domain names from the rule base and values from the universal set (input data)
            for which it is necessary to find the values of the consequences (defuzzified values of the output variable)

        Returns:
            Dict[str, float]: The numerical value of the consequence
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




