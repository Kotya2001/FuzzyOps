from typing import List
from fuzzyops.fuzzy_numbers import FuzzyNumber


def _is_dominated(check_solution: List[FuzzyNumber], main_solution: List[FuzzyNumber]) -> bool:
    """
    Checks whether the main solution dominates the tested solution by all criteria

    Args:
        check_solution (List[FuzzyNumber]): A solution that is checked for dominance
        main_solution (List[FuzzyNumber]): The main decision by which dominance is checked

    Returns:
        bool: True if the main solution dominates the one being tested, False otherwise
    """

    better_in_one_criterion = False
    for a, b in zip(check_solution, main_solution):
        if float(b) < float(a):
            return False
        if float(b) > float(a):
            better_in_one_criterion = True
    return better_in_one_criterion


def fuzzy_pareto_solver(solutions: List[List[FuzzyNumber]]) -> List[List[FuzzyNumber]]:
    """
    Finds solutions that are not dominated by other solutions in a multi-criteria problem

    Args:
        solutions (List[List[FuzzyNumber]]): A list of solutions, each of which is represented by a list of fuzzy numbers
        that are located on the Pareto boundary

    Returns:
        List[List[FuzzyNumber]]: A list of solutions that are not dominated by other solutions
        (there is at least one solution for which other solutions are better by all criteria)
    """

    pareto_solutions = []
    for solution in solutions:
        dominated = False
        for other_solution in solutions:
            if _is_dominated(solution, other_solution):
                dominated = True
                break
        if not dominated:
            pareto_solutions.append(solution)
    return pareto_solutions
