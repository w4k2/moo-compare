import numpy as np
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate

from config import *
from operator import itemgetter

table = []

for ds in DATASETS:
    X, y = DATASETS[ds]['data'], DATASETS[ds]['target']
    y = LabelEncoder().fit_transform(y)

    categorical_features = X.select_dtypes(include=['category']).columns.tolist()

    if len(categorical_features):
        print(f"{ds} has categorical features")

    n_samples = len(X)
    n_features = len(X.T)
    c_lab, c_num = np.unique(y, return_counts=True)
    ir = np.max(c_num) / np.min(c_num)
    table.append([ds, n_samples, n_features, ir, np.min(c_num)])

table = list(sorted(table, key=itemgetter(3)))

with open('datasets.tbl', 'w') as fp:
    fp.write(tabulate(table, tablefmt='grid', headers=['datasets', 'n_samples', 'n_features', 'ir', 'min_class_samples']))
    fp.write(f"\nTotal: {len(DATASETS)}")

with open(os.path.join("_tables", 'datasets.tex'), 'w') as fp:
    fp.write(tabulate(table, tablefmt='latex', headers=['datasets', 'n_samples', 'n_features', 'ir', 'min_class_samples']))
