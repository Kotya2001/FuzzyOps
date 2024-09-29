

def _is_dominated(check_solution, main_solution):
    """
    check
    """
    better_in_one_criterion = False
    for a, b in zip(check_solution, main_solution):
        if float(b) < float(a):
            return False
        if float(b) > float(a):
            better_in_one_criterion = True
    return better_in_one_criterion


def fuzzy_pareto_solver(solutions):
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