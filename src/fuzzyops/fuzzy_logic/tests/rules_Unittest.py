import unittest
import numpy as np
from src.fuzzyops.fuzzy_logic.base_rules import BaseRule, FuzzyInference
from src.fuzzyops.fuzzy_numbers import Domain

class TestFuzzyInference(unittest.TestCase):
    def setUp(self):
        age_domain = Domain((0, 100), name='age')
        age_domain.create_number('trapezoidal', -1, 0, 20, 30, name='young')
        age_domain.create_number('trapezoidal', 20, 30, 50, 60, name='middle')
        age_domain.create_number('trapezoidal', 50, 60, 100, 100, name='old')

        accident_domain = Domain((0, 1, 0.1), name='accident')
        accident_domain.create_number('trapezoidal', -0.1, 0., 0.1, 0.2, name='low')
        accident_domain.create_number('trapezoidal', 0.1, 0.2, 0.7, 0.8, name='medium')
        accident_domain.create_number('trapezoidal', 0.7, 0.8, 0.9, 1, name='high')

        self.domain = age_domain
        self.accident_domain = accident_domain

        self.ruleset = [
            BaseRule(age_domain.young, 'high'),
            BaseRule(age_domain.middle, 'medium'),
            BaseRule(age_domain.old, 'high'),
        ]

        self.fuzzy_inference = FuzzyInference(age_domain, self.ruleset)
    
    def test_fuzzy_inference_methods(self):
        self.fuzzy_inference.domain.method = 'minimax'
        self.domain.method = 'minimax'
        minimax_out = self.fuzzy_inference(25)
        self.assertTrue(np.allclose(minimax_out, 0.45000001788139343), 'Minimax method is not correct')

        self.fuzzy_inference.domain.method = 'prod'
        self.domain.method = 'prod'
        prod_out = self.fuzzy_inference(25)
        self.assertTrue(np.allclose(prod_out, 0.45000001788139343), 'Prod method is not correct')

    def test_fuzzy_inference_defuzz(self):
        self.fuzzy_inference.domain.method = 'minimax'
        self.domain.method = 'minimax'
        
        self.fuzzy_inference.defuzz_by = 'cgrav'
        outs = [self.fuzzy_inference(x) for x in range(0, 101, 5)]
        self.assertTrue(np.allclose(outs, [0.8500000238418579,
                                        0.8500000238418579,
                                        0.8500000238418579,
                                        0.8500000238418579,
                                        0.8500000238418579,
                                        0.45000001788139343,
                                        0.05000000074505806,
                                        0.05000000074505806,
                                        0.05000000074505806,
                                        0.05000000074505806,
                                        0.05000000074505806,
                                        0.45000001788139343,
                                        0.8500000238418579,
                                        0.8500000238418579,
                                        0.8500000238418579,
                                        0.8500000238418579,
                                        0.8500000238418579,
                                        0.8500000238418579,
                                        0.8500000238418579,
                                        0.8500000238418579,
                                        np.NaN]
                                        ), 'Defuzzification for cgrav is not correct')
        
        self.fuzzy_inference.domain.defuzz_by = 'cmax'
        outs = [self.fuzzy_inference(x) for x in range(0, 101, 5)]
        self.assertTrue(np.allclose(outs, [0.8500000238418579,
                                        0.8500000238418579,
                                        0.8500000238418579,
                                        0.8500000238418579,
                                        0.8500000238418579,
                                        0.44999998807907104,
                                        0.05000000074505806,
                                        0.05000000074505806,
                                        0.05000000074505806,
                                        0.05000000074505806,
                                        0.05000000074505806,
                                        0.44999998807907104,
                                        0.8500000238418579,
                                        0.8500000238418579,
                                        0.8500000238418579,
                                        0.8500000238418579,
                                        0.8500000238418579,
                                        0.8500000238418579,
                                        0.8500000238418579,
                                        0.8500000238418579,
                                        0.44999998807907104]
                                        ), 'Defuzzification for cmax is not correct')

        self.fuzzy_inference.domain.defuzz_by = 'lmax'
        outs = [self.fuzzy_inference(x) for x in range(0, 101, 5)]
        self.assertTrue(np.allclose(outs, [0.800000011920929,
                                        0.800000011920929,
                                        0.800000011920929,
                                        0.800000011920929,
                                        0.800000011920929,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.800000011920929,
                                        0.800000011920929,
                                        0.800000011920929,
                                        0.800000011920929,
                                        0.800000011920929,
                                        0.800000011920929,
                                        0.800000011920929,
                                        0.800000011920929,
                                        0.0]
                                        ), 'Defuzzification for lmax is not correct')

        self.fuzzy_inference.domain.defuzz_by = 'rmax'
        outs = [self.fuzzy_inference(x) for x in range(0, 101, 5)]
        self.assertTrue(np.allclose(outs, [0.8999999761581421,
                                        0.8999999761581421,
                                        0.8999999761581421,
                                        0.8999999761581421,
                                        0.8999999761581421,
                                        0.10000000149011612,
                                        0.10000000149011612,
                                        0.10000000149011612,
                                        0.10000000149011612,
                                        0.10000000149011612,
                                        0.10000000149011612,
                                        0.10000000149011612,
                                        0.8999999761581421,
                                        0.8999999761581421,
                                        0.8999999761581421,
                                        0.8999999761581421,
                                        0.8999999761581421,
                                        0.8999999761581421,
                                        0.8999999761581421,
                                        0.8999999761581421,
                                        0.10000000149011612]
                                        ), 'Defuzzification for lmax is not correct')
