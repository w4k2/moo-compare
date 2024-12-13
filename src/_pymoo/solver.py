from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination.default import DefaultMultiObjectiveTermination

from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation

from .operators import CallbackRepair
from .problem import PyMooCallbackProblem
from .output import ProgressOutput

ref_dirs = get_reference_directions("uniform", 2, n_partitions=2)

SOLVER_BASE = NSGA2
SOLVER_PARAMS = {
    "sampling": BinaryRandomSampling(),
    "crossover": PointCrossover(n_points=2, prob=0.2),
    "mutation": BitflipMutation(prob=0.9),
    "pop_size": 20
}

SOLVER_TERMINATION = DefaultMultiObjectiveTermination
SOLVER_TERMINATION_ARGS = {
    'n_max_gen': 500,
    # 'ftol': 0.0001,
    'xtol': 0.0001,
    'n_skip': 0,
    'period': 10,
}

PYMOO_VERBOSE = False
PYMOO_PROGRESS = True
PYMOO_HISTORY = False
RANDOM_STATE = 50310

def init_solver(callback, n_var, n_obj, repair_cb=None):
    repair = CallbackRepair(repair_cb) if repair_cb else None
    solver = SOLVER_BASE(repair=repair, **SOLVER_PARAMS)
    termination = SOLVER_TERMINATION(**SOLVER_TERMINATION_ARGS)
    solver.setup(PyMooCallbackProblem(callback, n_var, n_obj),
        termination=termination,
        seed=RANDOM_STATE,
        verbose=PYMOO_VERBOSE,
        save_history=PYMOO_HISTORY,
        progress=PYMOO_PROGRESS,
        display=ProgressOutput(),
    )

    return solver
