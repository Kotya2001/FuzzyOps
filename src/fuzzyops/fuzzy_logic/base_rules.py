from fuzzyops.fuzzy_numbers import Domain, FuzzyNumber

from typing import Union, Dict


class BaseRule:
    def __init__(self, antecedents, consequent):
        self.antecedents = antecedents
        self.consequent = consequent


class FuzzyInference:
    def __init__(self, domains, rules):
        self.domains = domains
        self.rules = rules

    def compute(self, input_data: Dict[str, Union[int, float, FuzzyNumber]]):
        results = {rule.consequent[0]: 0 for rule in self.rules}
        for rule in self.rules:
            antecedents = rule.antecedents
            consequent = rule.consequent
            res = 1
            cons_domain = self.domains.get(consequent[0])
            cons_ters = cons_domain.get(consequent[1])
            for antecedent in antecedents:
                domain_name = antecedent[0]
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
