from .fuzz_defuzzUnittest import TestSpeed, TestFuzzyNumber

import unittest
import argparse

test_speed = unittest.TestLoader().loadTestsFromTestCase(TestSpeed)
test_fuzzy_number = unittest.TestLoader().loadTestsFromTestCase(TestFuzzyNumber)

cases = {"speed": test_speed,
         "fn": test_fuzzy_number}


def run_tests():
    t = opt.type
    unittest.TextTestRunner(verbosity=2).run(cases[t])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test runner")
    parser.add_argument("type", type=str, help="speed, fn")
    opt = parser.parse_args()
    run_tests()
