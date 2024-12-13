from config import *

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from src.metrics import BinaryConfusionMatrix
from src.utils.moo import HV, GD
from src.pe_metrics import ParetoEfficiencyMetrics

RESAMPLING_RESULTS = os.path.join(RESULTS_DIR, "resampling_pred")
MOO_RESULTS = os.path.join(RESULTS_DIR, "moo_pred")

ed_scores = np.zeros((len(DATASETS), len(BASE_CLASSIFIERS), len(RESAMPLING), SPLIT.get_n_splits()))
hv_scores = np.zeros((len(DATASETS), len(BASE_CLASSIFIERS), len(RESAMPLING), SPLIT.get_n_splits()))
sdr_scores = np.zeros((len(DATASETS), len(BASE_CLASSIFIERS), len(RESAMPLING), SPLIT.get_n_splits()))
dr_scores = np.zeros((len(DATASETS), len(BASE_CLASSIFIERS), len(RESAMPLING), SPLIT.get_n_splits()))

for ds_i, ds_name in enumerate(DATASETS):
    X, y = DATASETS[ds_name]["data"], DATASETS[ds_name]["target"]
    y = LabelEncoder().fit_transform(y)

    for s_idx, (train, test) in enumerate(SPLIT.split(X, y)):

        for clf_i, _clf in enumerate(BASE_CLASSIFIERS):
            clf_name = type(_clf).__name__

            results = os.path.join(MOO_RESULTS, f"{ds_name}:{s_idx}:MEUS:{clf_name}.npy")
            y_pred = np.load(results)

            tpr, tnr = BinaryConfusionMatrix(y[test], y_pred).get_metrics("TPR", "TNR")
            pareto = np.stack([tpr, tnr]).T

            for ovs_i, _ovs in enumerate(RESAMPLING):
                if isinstance(_ovs, BaseEstimator):
                    osv_name = type(_ovs).__name__
                else:
                    osv_name = _ovs.__name__

                results = os.path.join(RESAMPLING_RESULTS, f"{ds_name}:{s_idx}:{osv_name}:{clf_name}.npy")
                y_pred = np.load(results)
                tpr, tnr = BinaryConfusionMatrix(y[test], y_pred).get_metrics("TPR", "TNR")

                pem = ParetoEfficiencyMetrics(pareto, np.array((tpr, tnr)))

                hv_scores[ds_i, clf_i, ovs_i, s_idx] = HV(pareto, (tpr, tnr))
                ed_scores[ds_i, clf_i, ovs_i, s_idx] = GD(pareto, (tpr, tnr))
                dr_scores[ds_i, clf_i, ovs_i, s_idx] = pem.dominance_ratio
                sdr_scores[ds_i, clf_i, ovs_i, s_idx] = pem.strict_dominance_ratio

np.save(os.path.join("_results", "hv_scores"), hv_scores)
np.save(os.path.join("_results", "ed_scores"), ed_scores)
np.save(os.path.join("_results", "dr_scores"), dr_scores)
np.save(os.path.join("_results", "sdr_scores"), sdr_scores)
