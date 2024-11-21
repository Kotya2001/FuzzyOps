from typing import List
from fuzzyops.fuzzy_numbers import FuzzyNumber


def _is_dominated(check_solution: List[FuzzyNumber], main_solution: List[FuzzyNumber]) -> bool:
    """
    Проверяет, доминирует ли основное решение по всем критериям над проверяемым решением.

    Args:
        check_solution (List[FuzzyNumber]): Решение, которое проверяется на доминирование.
        main_solution (List[FuzzyNumber]): Основное решение, по которому проверяется доминирование.

    Returns:
        bool: True, если основное решение доминирует над проверяемым, False в противном случае.
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
    Находит решения, не доминируемые другими решениями в многокритериальной задаче.

    Args:
        solutions (List[List[FuzzyNumber]]): Список решений, каждое из которых представлено списком нечетких чисел.

    Returns:
        List[List[FuzzyNumber]]: Список решений, которые не доминируются другими решениями
        (существует хотя бы одно решение, по которому другие решения лучше по всем критериям).
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
