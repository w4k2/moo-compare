from pymoo.core.problem import Problem
import numpy as np

class PyMooCallbackProblem(Problem):
    def __init__(self, eval_cb, n_var, n_obj, n_constr=0, vtype=bool):
        self.eval_cb = eval_cb

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, vtype=vtype)

    def _calc_pareto_front(self, n_pareto_points=1):
        return np.array([[-1.0, -1.0]])

    def _evaluate(self, S, out, *args, **kwargs):
        out["F"] = self.eval_cb(S)
