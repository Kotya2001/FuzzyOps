from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import torch


class FuzzyProblem(Problem):
    def __init__(self, n_vars, low, high, crits, bounds, task_type="min"):
        self.fs = [crit.membership for crit in crits]
        self.gs = [bound.membership for bound in bounds]
        self.task_type = task_type
        super().__init__(n_var=n_vars, n_obj=len(crits), n_constr=len(bounds), xl=low, xu=high)

    def _evaluate(self, x, out, *args, **kwargs):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        # Вычисление целевых функций
        f = torch.stack(
            [torch.stack([func(x_tensor[:, i]) for i in range(x_tensor.shape[1])], dim=1) for func in self.fs],
            dim=2)

        if self.task_type == "min":
            f = torch.mean(f, dim=1)
        else:
            f = -torch.mean(f, dim=1)

        g = torch.stack(
            [torch.stack([func(x_tensor[:, i]) for i in range(x_tensor.shape[1])], dim=1) for func in self.gs],
            dim=2)
        g = torch.mean(g, dim=1)

        out["F"] = f.cpu().numpy()
        out["G"] = g.cpu().numpy()


def solve_multy_obj_task(n_vars, low, high, crits, bounds, n_gen, verbose=True, seed=1, task_type="min"):
    algorithm = NSGA2()
    problem = FuzzyProblem(n_vars, low, high, crits, bounds, task_type=task_type)
    res = minimize(problem,
                   algorithm,
                   ('n_gen', n_gen),
                   seed=seed,
                   verbose=verbose)

    return res.X, res.F
