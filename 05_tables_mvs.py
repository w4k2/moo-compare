from config import *

import numpy as np
from tabulate import tabulate
from sklearn.base import BaseEstimator

CLF = 0

ovs_names = []
for ovs_i, _ovs in enumerate(RESAMPLING):
    if isinstance(_ovs, BaseEstimator):
        osv_name = type(_ovs).__name__
    else:
        osv_name = _ovs.__name__

    ovs_names.append(osv_name)

# len(DATASETS), len(BASE_CLASSIFIERS), len(RESAMPLING), SPLIT.get_n_splits()
METRICS = [
    ("ED", np.load(os.path.join("_results", "ed_scores.npy")), lambda x: f"{x:.2f}"),
    ("HV", np.load(os.path.join("_results", "hv_scores.npy")), lambda x: f"{x * 1000:.2f}"),
    ("DR", np.load(os.path.join("_results", "dr_scores.npy")), lambda x: f"{x:.2f}"),
    ("SDR", np.load(os.path.join("_results", "sdr_scores.npy")), lambda x: f"{x:.2f}"),
]

for m_name, m_table, m_str in METRICS:
    mean_v = np.mean(m_table, axis=-1)[:, CLF, :].T
    std_v = np.std(m_table, axis=-1)[:, CLF, :].T

    index = np.array(ovs_names)
    header = ['Sampling'] + [f'$\\mathcal{{DS}}_{{{_}}}$' for _ in range(1, len(DS_NAMES) + 1)]

    mean_s = np.vectorize(lambda x: f"\makecell{{{m_str(x)} \\\\")(mean_v)
    std_s = np.vectorize(lambda x: f" \\tiny{{ \\color{{gray}} ({m_str(x)})}}}}")(std_v)
    table_data = np.char.add(mean_s, std_s)
    table_data = np.concatenate([index[:, None], table_data], axis=1)
    table = tabulate(table_data, tablefmt="latex_raw", headers=header)

    with open(os.path.join("_tables", f'{m_name}.tex'), 'w') as fp:
        fp.write(table)

    mean_s = np.vectorize(lambda x: f"{m_str(x)}")(mean_v)
    std_s = np.vectorize(lambda x: f"\n({m_str(x)})")(std_v)
    table_data = np.char.add(mean_s, std_s)
    table_data = np.concatenate([index[:, None], table_data], axis=1)
    table = tabulate(table_data, tablefmt="grid", headers=header)

    with open(os.path.join("_tables", f'{m_name}.tbl'), 'w') as fp:
        fp.write(table)