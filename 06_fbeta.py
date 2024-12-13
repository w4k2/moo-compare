import numpy as np
import matplotlib.pyplot as plt
import os
from config import *

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from src.metrics import BinaryConfusionMatrix
from functools import partial

betas = np.geomspace(0.1, 10, 200)


def f_beta(TPR, PPV, beta=1.0):
    if not hasattr(beta, "__iter__"):
        beta = np.array([beta])

    beta_sqr = np.power(beta, 2)

    return np.nan_to_num(
        (1 + beta_sqr)
        * PPV[..., np.newaxis]
        * TPR[..., np.newaxis]
        / ((beta_sqr * PPV[..., np.newaxis]) + TPR[..., np.newaxis])
    )


def f_beta_zero_point(TPR_A, PPV_A, TPR_B, PPV_B):
    return np.sqrt(
        (TPR_A * TPR_B * (PPV_B - PPV_A)) / (PPV_A * PPV_B * (TPR_A - TPR_B))
    )


RESAMPLING_RESULTS = os.path.join(RESULTS_DIR, "resampling_pred")
MOO_RESULTS = os.path.join(RESULTS_DIR, "moo_pred")

for ds_i, ds_name in enumerate(DATASETS):
    X, y = DATASETS[ds_name]["data"], DATASETS[ds_name]["target"]
    y = LabelEncoder().fit_transform(y)

    for clf_i, _clf in enumerate(BASE_CLASSIFIERS):
        clf_name = type(_clf).__name__

        for s_idx, (train, test) in enumerate(SPLIT.split(X, y)):

            results = os.path.join(
                MOO_RESULTS, f"{ds_name}:{s_idx}:MEUS:{clf_name}.npy"
            )
            y_pred = np.load(results)

            pareto_tpr, pareto_ppv = BinaryConfusionMatrix(y[test], y_pred).get_metrics(
                "TPR", "PPV"
            )

            ref_tpr, ref_ppv = [], []
            ovs_names = []
            for ovs_i, _ovs in enumerate(RESAMPLING):
                if isinstance(_ovs, BaseEstimator):
                    osv_name = type(_ovs).__name__
                else:
                    osv_name = _ovs.__name__

                ovs_names.append(osv_name)

                results = os.path.join(
                    RESAMPLING_RESULTS, f"{ds_name}:{s_idx}:{osv_name}:{clf_name}.npy"
                )
                y_pred = np.load(results)
                tpr, ppv = BinaryConfusionMatrix(y[test], y_pred).get_metrics(
                    "TPR", "PPV"
                )
                ref_tpr.append(tpr)
                ref_ppv.append(ppv)

            ref_tpr = np.array(ref_tpr)
            ref_ppv = np.array(ref_ppv)

            K = 4
            # fig, axs = plt.subplots(
            #     1, 2, figsize=(K * 2, K), gridspec_kw={"width_ratios": [1, 3]}
            # )

            fig, ax = plt.subplots(
                1, 1, figsize=(K * 2, K)
            )

            f_betas = f_beta(ref_tpr, ref_ppv, beta=betas).T
            best = np.argmax(f_betas, axis=-1)
            a = np.pad(best[1:], (0, 1), "edge") - best
            best_map = a != 0
            best_map[-1] = True
            unique_best = best[best_map]

            pareto_f_betas = f_beta(pareto_tpr, pareto_ppv, beta=betas).T
            pareto_best = np.argmax(pareto_f_betas, axis=-1)
            a = np.pad(pareto_best[1:], (0, 1), "edge") - pareto_best
            pareto_best_map = a != 0
            pareto_best_map[-1] = True
            pareto_unique_best = pareto_best[pareto_best_map]

            # ax = axs[0]

            # ax.grid(ls=":")
            # ax.set_xlim(0.0, 1.0)
            # ax.set_ylim(0.0, 1.0)
            # ax.set_xscale("function", functions=(partial(np.power, 10.0), np.log10))
            # ax.set_yscale("function", functions=(partial(np.power, 10.0), np.log10))
            # ax.set_aspect("equal")
            # ax.set_xlabel("TPR")
            # ax.set_ylabel("PPV")
            # ax.set_xticks([0.0, 0.5, 0.8, 1.0])
            # ax.set_yticks([0.0, 0.5, 0.8, 1.0])
            # ax.spines[["right", "top"]].set_visible(False)

            # ax.scatter(ref_tpr, ref_ppv, s=40, c="#DDDDDD")
            # ax.scatter(pareto_tpr, pareto_ppv, s=40, marker="D", c="#DDDDDD")

            # for b in unique_best:
            #     ax.scatter(ref_tpr[b], ref_ppv[b], s=70, marker="o")

            # for b in pareto_unique_best:
            #     ax.scatter(pareto_tpr[b], pareto_ppv[b], s=40, marker="D", c="k")

            # ax = axs[1]

            ax.grid(ls=":")
            ax.set_xscale("log")
            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(np.min(betas), np.max(betas))
            ax.set_ylabel("$F_\\beta$ score")

            # if n == n_splits - 1:
            ax.set_xlabel("$\\beta$")

            ax.spines[["right", "top"]].set_visible(False)

            ax.plot(betas, f_betas, c="#DDDDDD", lw=1)

            for b in unique_best:
                ax.plot(
                    betas[best == b], f_betas[:, b][best == b], lw=3, label=ovs_names[b]
                )[0]

            best_iter = iter(unique_best)
            last = next(best_iter)

            # while x := next(best_iter, None):
            #     a = f_beta_zero_point(
            #         ref_tpr[last], ref_ppv[last], ref_tpr[x], ref_ppv[x]
            #     )
            #     ax.vlines(a, 0.0, 1.0, color="k", lw=1, ls=":")
            #     ax.text(
            #         a,
            #         0.92,
            #         f"{a:.2f}",
            #         rotation=90,
            #         fontsize="smaller",
            #         horizontalalignment="right",
            #     )

            #     last = x

            for b in pareto_unique_best:
                ax.plot(
                    betas[pareto_best == b],
                    pareto_f_betas[:, b][pareto_best == b],
                    c="w",
                    lw=5,
                )
                ax.plot(
                    betas[pareto_best == b],
                    pareto_f_betas[:, b][pareto_best == b],
                    c="k",
                    ls="--",
                    lw=2,
                )

            ax.plot(
                betas[pareto_best == b],
                pareto_f_betas[:, b][pareto_best == b],
                c="k",
                ls="--",
                lw=2,
                label="MEUS",
            )

            ax.legend(
                loc="lower left",
                bbox_to_anchor=(0.025, 0.025),
                fancybox=False,
                ncol=2,
                prop={"size": "large"},
            )

            plt.tight_layout()
            plt.savefig(f"_plots/fbeta:{ds_name}:{s_idx}:{clf_name}.png")
            plt.savefig(f"_plots/fbeta:{ds_name}:{s_idx}:{clf_name}.pdf")
            plt.savefig("foo.png")
            plt.clf()
            plt.close()
            break
