from config import *

import os
import numpy as np

from smote_variants.base import RandomSamplingMixin, OverSampling
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import clone, BaseEstimator

from tqdm import tqdm

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

    bar = tqdm(desc=ds_name, total=SPLIT.get_n_splits(), leave=False)

    for s_idx, (train, test) in enumerate(SPLIT.split(X, y)):
        scaler = StandardScaler()
        X_scale = scaler.fit_transform(X[train])

        bar_sampling = tqdm(RESAMPLING, leave=False)

        for o_idx, _ovs in enumerate(bar_sampling):
            if isinstance(_ovs, BaseEstimator):
                osv_name = type(_ovs).__name__
                ovs = clone(_ovs)
            else:
                osv_name = _ovs.__name__
                ovs = _ovs()

                if isinstance(ovs, RandomSamplingMixin):
                    ovs.set_random_state(RANDOM_STATE)

            bar_sampling.set_description(osv_name)

            X_train, y_train = ovs.fit_resample(X_scale, y[train])

            for c_idx, _clf in enumerate(BASE_CLASSIFIERS):
                clf_name = type(_clf).__name__

                results_path = os.path.join(RESULTS_DIR, "resampling_pred", f"{ds_name}:{s_idx}:{osv_name}:{clf_name}")
                if SKIP_EXISTING_PREDICTIONS and os.path.exists(f"{results_path}.npy"):
                    continue

                clf = clone(_clf)
                clf.fit(X_train, y_train)

                X_test = scaler.transform(X[test])

                if isinstance(ovs, OverSampling) and OverSampling.cat_dim_reduction in ovs.categories:
                    X_test = ovs.preprocessing_transform(X_test)

                y_pred = clf.predict(X_test)
                np.save(results_path, y_pred)
