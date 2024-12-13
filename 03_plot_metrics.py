from config import *

import numpy as np
from sklearn.base import BaseEstimator
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from src.metrics import BinaryConfusionMatrix
from functools import partial
from adjustText import adjust_text

FR = 5

def get_scatter_style():
    for m in []:
        for c in []:
            yield m, c

RESAMPLING_RESULTS = os.path.join(RESULTS_DIR, "resampling_pred")
MOO_RESULTS = os.path.join(RESULTS_DIR, "moo_pred")

for ds_name in DATASETS:
    X, y = DATASETS[ds_name]["data"], DATASETS[ds_name]["target"]
    y = LabelEncoder().fit_transform(y)

    for s_idx, (train, test) in enumerate(SPLIT.split(X, y)):
        for _clf in BASE_CLASSIFIERS:
            fig, ax = plt.subplots(1, 1, figsize=(FR, FR))

            clf_name = type(_clf).__name__

            results = os.path.join(MOO_RESULTS, f"{ds_name}:{s_idx}:MEUS:{clf_name}.npy")
            y_pred = np.load(results)

            tpr, tnr = BinaryConfusionMatrix(y[test], y_pred).get_metrics("TPR", "TNR")
            ax.scatter(tpr, tnr, c='r', s=8)
            ax.set_aspect('equal')

            texts = []

            for _ovs in RESAMPLING:
                if isinstance(_ovs, BaseEstimator):
                    osv_name = type(_ovs).__name__
                else:
                    osv_name = _ovs.__name__

                results = os.path.join(RESAMPLING_RESULTS, f"{ds_name}:{s_idx}:{osv_name}:{clf_name}.npy")
                y_pred = np.load(results)

                tpr, tnr = BinaryConfusionMatrix(y[test], y_pred).get_metrics("TPR", "TNR")
                ax.scatter(tpr, tnr, c='k', s=6)
                texts.append(ax.text(tpr, tnr, osv_name, fontsize=6))
                ax.grid(ls=":")
                ax.set_xscale('function', functions=(partial(np.power, 10.0), np.log10))
                ax.set_yscale('function', functions=(partial(np.power, 10.0), np.log10))
                ax.set_aspect('equal')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

            adjust_text(texts, avoid_self=False, adjust=True, expand=(2, 2),
            arrowprops=dict(arrowstyle='->', color='gray', alpha=.5))

            print(f"{ds_name}-{s_idx}-{clf_name}")

            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"{ds_name}-{s_idx}-{clf_name}.png"))
            plt.savefig(os.path.join(PLOTS_DIR, f"{ds_name}-{s_idx}-{clf_name}.pdf"))
            plt.savefig("foo.png")


            plt.close()
            plt.clf()
