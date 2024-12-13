import numpy as np
from scipy.spatial.distance import cdist


def non_dominated_solutions(pareto, distinct=False):
    is_efficient = np.zeros(len(pareto), dtype=bool)

    for i in range(len(pareto)):
        this_cost = pareto[i, :]

        at_least_as_good = np.logical_not(np.any(pareto < this_cost, axis=1))
        any_better = np.any(pareto > this_cost, axis=1)

        dominated_by = np.logical_and(at_least_as_good, any_better)

        if distinct and np.any(is_efficient):
            if np.any(np.all(pareto[is_efficient] == this_cost, axis=1)):
                continue

        if not (np.any(dominated_by[:i]) or np.any(dominated_by[i + 1 :])):
            is_efficient[i] = True

    return is_efficient


def GD(pareto, ideal, p=2.0):
    ideal = np.array(ideal)
    if len(ideal.shape) == 1:
        ideal = np.array([ideal])

    dist_mat = cdist(pareto, ideal, metric='minkowski', p=2.0)
    return np.mean(np.min(dist_mat, axis=1))


def HV(front, ref_point):
    dominating = np.all(front > ref_point, axis=1)
    d_front = front[dominating]
    p_dominating = non_dominated_solutions(d_front)
    d_front = d_front[p_dominating]
    sorted_front = d_front[np.argsort(d_front[:, 1])]

    areas = []
    for a, b in zip(np.vstack([ref_point, sorted_front]), sorted_front):
        areas.append((ref_point[0] - b[0]) * (a[1] - b[1]))

    return np.sum(areas)
