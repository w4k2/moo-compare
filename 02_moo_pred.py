from config import *

import os
import numpy as np

from src.meus import MEUS
from smote_variants.base import RandomSamplingMixin, OverSampling
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import clone, BaseEstimator

from tqdm import tqdm

RESULTS = "moo_pred"

for ds_name in DATASETS:
    X, y = DATASETS[ds_name]["data"], DATASETS[ds_name]["target"]

    categorical_features = X.select_dtypes(include=['category']).columns.tolist()

    if len(categorical_features):
        print(f"{ds_name} has categorical features")

    encoder = OneHotEncoder(sparse_output=False)
    ohe = ColumnTransformer(
        transformers=[("ohe", encoder, categorical_features),],
        remainder="passthrough",
    )

    X = ohe.fit_transform(X)
    X.astype(float)

    y = LabelEncoder().fit_transform(y)

    bar = tqdm(desc=ds_name, total=SPLIT.get_n_splits(), leave=False, position=2)

    for s_idx, (train, test) in enumerate(SPLIT.split(X, y)):
        scaler = StandardScaler()
        X_scale = scaler.fit_transform(X[train])
        X_train, y_train = X_scale, y[train]

        bar_classifiers = tqdm(BASE_CLASSIFIERS, leave=False, position=1)
        for c_idx, _clf in enumerate(BASE_CLASSIFIERS):
            clf_name = type(_clf).__name__

            results_path = os.path.join(RESULTS_DIR, "moo_pred", f"{ds_name}:{s_idx}:MEUS:{clf_name}")
            if SKIP_EXISTING_PREDICTIONS and os.path.exists(f"{results_path}.npy"):
                continue

            clf = clone(_clf)
            meus = MEUS(estimator=clf)
            meus.fit(X_train, y_train)

            X_test = scaler.transform(X[test])
            y_pred = meus.predict(X_test)
            np.save(results_path, y_pred)

            bar_classifiers.update()

    bar.update()