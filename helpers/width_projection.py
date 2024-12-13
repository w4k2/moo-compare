import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

X = np.array(
    [[0.2, 0.95], [0.3, 0.91], [0.35, 0.9], [0.72, 0.8], [0.85, 0.7], [0.9, 0.45]]
)

Z = X[:, 0] - X[:, 1]
Q = (X[:, 0] + X[:, 1]) / 2

c1 = (1 - Z) / 2
c0 = -c1 + 1

G = np.vstack([Q, Q]).T
C = np.vstack([c0, c1]).T

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.grid(ls=':', zorder=0)

ax.plot((-0.00, 1.00), (1.00, -0.00), ls='--', lw=1, c='grey', zorder=1)
ax.plot((1.00, 0.00), (1.00, 0.00), ls='--', lw=1, c='grey', zorder=1)

ax.add_patch(
    patches.Rectangle((0, 0), 1, 1, ec="k", lw=1, ls=':', fill=None, zorder=2),
)

ax.add_patch(
    patches.Polygon([[0, 0], [0, 1], [1, 0]], facecolor="k", alpha=0.05)
)

ax.scatter(*C.T, c='darkorange', s=15)
ax.scatter(*G.T, c='royalblue', s=15)

for a, b in zip(X, C):
    ax.plot((a[0], b[0]), (a[1], b[1]), ls=':', c='darkorange')

for a, b in zip(X, G):
    ax.plot((a[0], b[0]), (a[1], b[1]), ls=':', c='royalblue')

ax.scatter(*X.T, c='maroon', marker='D', s=70, zorder=5)
ax.scatter(*X.T, c='ivory', marker='D', s=20, zorder=6)

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.spines[['right', 'top']].set_visible(False)

plt.tight_layout()
plt.savefig("foo.png")
plt.close()
plt.clf()
