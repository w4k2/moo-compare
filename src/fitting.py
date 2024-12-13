from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, RepeatedKFold, RepeatedStratifiedKFold
import numpy as np


class HoldOutFitting:
    def __init__(self, test_size=0.2, stratified=True, random_state=None):
        self.test_size = test_size
        self.stratified = stratified
        self.random_state = random_state

    def initialize(self, X, y):
        self.X = X
        self.y = y

        split = (
            StratifiedShuffleSplit(
                n_splits=1, test_size=self.test_size, random_state=self.random_state
            )
            if self.stratified
            else ShuffleSplit(
                n_splits=1, test_size=self.test_size, random_state=self.random_state
            )
        )

        self.train, self.test = next(split.split(X, y))
        return self.train

    def update(self):
        pass # Does not support changes of test set

    def __iter__(self):
        yield np.arange(len(self.train)), self.X[self.test], self.y[self.test]


class DummyFitting:
    def __init__(self, test_size=2.0, random_state=None):
        self.test_size = test_size
        self.random_state = random_state

        self._rs = (
            self.random_state
            if isinstance(self.random_state, np.random.RandomState)
            else np.random.RandomState(self.random_state)
        )

    def initialize(self, X, y):
        self.X = X
        self.y = y

        self.test = self.make_test()

        return np.arange(len(self.X))

    def make_test(self):
        if self.test_size >= 1.0:
            return np.arange(len(self.X))

        return self._rs.choice(
            len(self.X), size=int(len(self.X) * self.test_size), replace=False
        )

    def update(self):
        self.test = self.make_test()

    def __iter__(self):
        yield np.arange(len(self.X)), self.X[self.test], self.y[self.test]


class BootstrapFitting:
    def __init__(self, bootstrap_size=1.0, shrinkage=1.0, random_state=None):
        self.bootstrap_size = bootstrap_size
        self.shrinkage = shrinkage
        self.random_state = random_state

        self._rs = (
            self.random_state
            if isinstance(self.random_state, np.random.RandomState)
            else np.random.RandomState(self.random_state)
        )

    def make_boostrap(self):
        strategy = {
            cl: c_size + int(c_size * self.bootstrap_size)
            for cl, c_size in Counter(self.y).items()
        }

        X_, y_ = RandomOverSampler(
            sampling_strategy=strategy, shrinkage=self.shrinkage, random_state=self._rs
        ).fit_resample(self.X, self.y)

        # Drop original samples
        return X_[len(self.X):], y_[len(self.X):]

    def initialize(self, X, y):
        self.X = X
        self.y = y
        self.bootstrap = self.make_boostrap()

        return np.arange(len(self.X))

    def update(self):
        self.bootstrap = self.make_boostrap()

    def __iter__(self):
        yield np.arange(len(self.X)), *self.bootstrap


class CrossValidationFitting:
    def __init__(self, n_repeats=1, n_splits=5, stratified=True, random_state=None):
        self.n_repeats = n_repeats
        self.n_splits = n_splits
        self.stratified = stratified
        self.random_state = random_state

        self._rs = (
            self.random_state
            if isinstance(self.random_state, np.random.RandomState)
            else np.random.RandomState(self.random_state)
        )

    def initialize(self, X, y):
        self.X = X
        self.y = y
        self.cv = self.make_cv()

        return np.arange(len(self.X))

    def make_cv(self):
        cv_class = RepeatedStratifiedKFold if self.stratified else RepeatedKFold
        seed = self._rs.randint(0xffffffff) # Generate new seed, as it will assure same split if not dynamic
        return cv_class(n_repeats=self.n_repeats, n_splits=self.n_splits, random_state=seed)

    def update(self):
        self.cv = self.make_cv()

    def __iter__(self):
        for train, test in self.cv.split(self.X, self.y):
            yield train, self.X[test], self.y[test]
