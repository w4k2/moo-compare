import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
import numpy as np
from functools import partial

# plt.rcParams.update({
#     "text.usetex": True,
# })

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

colors = []

cm = np.array([
    [[177, 173, 96], [66, 121, 97]],
    [[140, 104, 89],[95, 69, 119]]
])

def efficiency_matrix(front, ref, weak=False):
    cmp = (front >= ref).T if weak else (front > ref).T
    cnt = np.bincount(2 * cmp[0] + cmp[1], minlength=4)
    return cnt.reshape(2, 2)

X = np.array([
    [0.2, 0.95], [0.3, 0.91], [0.35, 0.9], [0.72, 0.8], [0.85, 0.7], [0.9, 0.45]
])

N = 10
mean = [0.63, 0.78]
cov = [[0.001, -0.01],[-0.01, 0.001]]
seed = np.random.randint(2000000)
seed = 1268661
print(seed)
np.random.seed(seed)
X = np.random.multivariate_normal(mean, cov, N)

ref = np.array([0.60, 0.75])

em = efficiency_matrix(X, ref)
print(em)

# Z = X[:, 0] - X[:, 1]
# Q = (X[:, 0] + X[:, 1]) / 2

# c1 = (1 - Z) / 2
# c0 = -c1 + 1

# G = np.vstack([Q, Q]).T
# C = np.vstack([c0, c1]).T

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.grid(ls=':', zorder=1)

# ax.add_patch(
#     patches.Rectangle((0, 0), 1, 1, ec="k", lw=1, ls='-', fill=None, zorder=2),
# )

# for a in [[0, 0], [0, 1], [1, 1], [1, 0]]:

# patches.Polygon([[0, 0], [0, 1], [1, 1], [1, 0]], facecolor=cm[0, 0] / 255)

ax.scatter(*X.T, c='k', marker='o', s=30, zorder=3)
ax.scatter(*X.T, c='w', marker='o', s=5, zorder=4)

ax.scatter(*ref, c='k', marker='D', s=50, zorder=5)
ax.scatter(*ref, c='r', marker='D', s=10, zorder=6)

ax.vlines(ref[0], 0.0, 1.0, color="k", lw=1, ls=":")
ax.hlines(ref[1], 0.0, 1.0, color="k", lw=1, ls=":")

ax.set_xscale('function', functions=(partial(np.power, 10.0), np.log10))
ax.set_yscale('function', functions=(partial(np.power, 10.0), np.log10))

ax.text(0.35, 0.5, '$p \prec r $', horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='white', edgecolor='black',linewidth=1), fontsize=10)
ax.text(0.8, 0.90, '$p \succ r $', horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='white', edgecolor='black',linewidth=1), fontsize=10)
ax.text(0.35, 0.90, '$p_{TNR} > r_{TNR}$', horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='white', edgecolor='black',linewidth=1), fontsize=10)
ax.text(0.8, 0.5, '$p_{TPR} > r_{TPR}$', horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='white', edgecolor='black',linewidth=1), fontsize=10)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("TPR")
ax.set_ylabel("TNR")
ax.spines[['right', 'top']].set_visible(False)

# sub_ax = inset_axes(
#     parent_axes=ax,
#     width="20%",
#     height="20%",
#     loc='lower left'
# )

# ax = sub_ax

# im = sub_ax.imshow(np.zeros_like(em), cmap='gray_r')

# em = np.array([
#     [em[0, 1], em[1, 1]],
#     [em[0, 0], em[1, 0]],
# ])

# for i in range(em.shape[0]):
#     for j in range(em.shape[1]):
#         text = ax.text(j, i, em[i, j], ha="center", va="center", color="k")

# ax.vlines(0.5, -0.5, 1.5, color="k", lw=1, ls="-")
# ax.hlines(0.5, -0.5, 1.5, color="k", lw=1, ls="-")
# ax.spines[['right', 'top', 'bottom', 'left']].set_linewidth(2)

# ax.set_xticks([])
# ax.set_yticks([])
# ax.grid(ls='-', c='w')


plt.tight_layout()
plt.savefig("foo.png")
plt.savefig("foo.pdf")
plt.close()
plt.clf()


fig, axs = plt.subplots(1, 1, figsize=(5, 5))

ax = axs
ax.scatter(*X.T, c='gray')

ax.scatter(*ref, c='k', marker='D', s=50, zorder=5)
ax.scatter(*ref, c='r', marker='D', s=10, zorder=6)

ax.scatter(*X.T, c='k', marker='o', s=30, zorder=3)
ax.scatter(*X.T, c='w', marker='o', s=5, zorder=4)

ax.set_aspect('equal')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(ls=':')

areas = []
dominating = np.all(X > ref, axis=1)
d_front = X[dominating]
p_dominating = non_dominated_solutions(d_front)
d_front = d_front[p_dominating]

ax.set_xscale('function', functions=(partial(np.power, 10.0), np.log10))
ax.set_yscale('function', functions=(partial(np.power, 10.0), np.log10))
ax.scatter(*d_front.T, c="r", marker="D", s=45)
ax.set_xlabel("TPR")
ax.set_ylabel("TNR")

for d in d_front:
    diff = d - ref
    ax.add_patch(
        patches.Rectangle(ref, *diff, ec="gray", lw=0, facecolor="gray", zorder=-1),
    )

ax.spines[['right', 'top']].set_visible(False)

plt.tight_layout()
plt.savefig("bar.png")
plt.savefig("bar.pdf")
plt.close()
plt.clf()
