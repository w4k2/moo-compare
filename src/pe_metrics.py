import numpy as np
from functools import cached_property

# Efficiency Matrix
#
# TODO: Add minimalization option
#
# shape (2, 2) where
# (0, 0) - strictly dominated solutions, (1, 1) - strictly dominating solutions
# and rest are dominating solutions.
#
def efficiency_matrix(front, ref, weak=False):
    cmp = (front >= ref).T if weak else (front > ref).T
    cnt = np.bincount(2 * cmp[0] + cmp[1], minlength=4)
    return cnt.reshape(2, 2)

class ParetoEfficiencyMetrics:
    def __init__(self, front, ref):
        self.dm = efficiency_matrix(front, ref)

    @cached_property
    def strict_dominance_ratio(self):
        return self.dm[1, 1] / np.sum(self.dm)

    @cached_property
    def dominance_ratio(self):
        return 1 - (self.dm[0, 0] / np.sum(self.dm))
