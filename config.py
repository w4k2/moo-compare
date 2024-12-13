import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

from imblearn.under_sampling import *
from imblearn.over_sampling import *
from smote_variants.oversampling import *


# Disable sv logging
import logging

logger = logging.getLogger("smote_variants")
logger.disabled = True

RANDOM_STATE = 50310
SPLIT = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=RANDOM_STATE)
SKIP_EXISTING_PREDICTIONS = True

OVERSAMPLING = [
    RandomOverSampler,
    SMOTE,
    ProWSyn,
    BorderlineSMOTE,
    DBSMOTE,
    SOMO,
    MSYN,
    CCR,
    AHC,
    ADASYN,
]

UNDERSAMPLING = [
    RandomUnderSampler(random_state=RANDOM_STATE),
    ClusterCentroids(random_state=RANDOM_STATE),
    InstanceHardnessThreshold(random_state=RANDOM_STATE),
    NearMiss,
    TomekLinks,
    EditedNearestNeighbours,
    AllKNN,
    OneSidedSelection(random_state=RANDOM_STATE),
    CondensedNearestNeighbour(random_state=RANDOM_STATE),
    NeighbourhoodCleaningRule,
]

RESAMPLING = [NoSMOTE] + OVERSAMPLING + UNDERSAMPLING

# Datasets
from src.utils.keel import parse_keel_dat

# DS_NAMES = [ds_fname.split('.')[0] for ds_fname in os.listdir('datasets/keel')]

DS_NAMES = [
    "vehicle1",
    "segment0",
    "yeast-0-2-5-6_vs_3-7-8-9",
    "shuttle-c0-vs-c4",
    "ecoli4",
    "pima",
    "page-blocks0",
    "winequality-white-3_vs_7",
    "new-thyroid1",
    "poker-8-9_vs_5",
]

DATASETS = {
    relation: {'data': data, 'target': target} for relation, data, target in
    [parse_keel_dat(os.path.join('datasets', 'keel', f"{ds_name}.dat")) for ds_name in DS_NAMES]
}

RESULTS_DIR = "_results"
PLOTS_DIR = "_plots"

BASE_CLASSIFIERS = [
    GaussianNB(),
    # KNeighborsClassifier(),
    # DecisionTreeClassifier(random_state=RANDOM_STATE),
    # MLPClassifier(random_state=RANDOM_STATE, tol=0.005, max_iter=100),
]
