from fuzz_defuzzUnittest import TestFuzzyNumber, TestSpeed

import unittest


test_speed = unittest.TestLoader().loadTestsFromTestCase(TestSpeed)
test_fuzzy_number = unittest.TestLoader().loadTestsFromTestCase(TestFuzzyNumber)

cases = {"speed": test_speed,
         "fn": test_fuzzy_number}


def run_fn_test():
    unittest.TextTestRunner(verbosity=2).run(cases["fn"])


def run_speed_test():
    unittest.TextTestRunner(verbosity=2).run(cases["speed"])


run_fn_test()
run_speed_test()