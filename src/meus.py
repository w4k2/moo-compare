import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
from tqdm import tqdm

from ._pymoo import init_solver
from .fitting import DummyFitting
from .metrics import BinaryConfusionMatrix

DEFAULT_ESTIMATOR = GaussianNB()
DEFAULT_FITTING = DummyFitting()
DEFAULT_METRICS = ["PPV", "NPV"]
DYNAMIC = False

class MEUS:
    def __init__(self, estimator=DEFAULT_ESTIMATOR, fitting=DEFAULT_FITTING, metrics=DEFAULT_METRICS):
        self.estimator = estimator
        self.fitting = fitting
        self.metrics = metrics

        self._valid_X_y = None
        self._training_history = []
        self._models = []

    def set_valid_X_y(self, X, y):
        self._valid_X_y = (X, y)
        self._validation_history = []

    def fit(self, X, y):
        self._classes = np.unique(y)
        self.train_ind = self.fitting.initialize(X, y)

        solver = init_solver(
            callback=self._eval_cb(X[self.train_ind], y[self.train_ind]),
            n_var=len(self.train_ind),
            n_obj=len(self.metrics),
            repair_cb=self._repair_cb(X[self.train_ind], y[self.train_ind]),
        )

        while solver.has_next():
            solver.next()

            self._training_history.append(
                -1 * solver.pop.get("F")
            )

            if self._valid_X_y:
                X_valid, y_valid = self._valid_X_y
                valid_scores = []

                for s in solver.pop.get("x"):
                    clf = clone(self.estimator)
                    clf.fit(X[self.train_ind[s]], y[self.train_ind[s]])
                    y_pred = clf.predict(X_valid)
                    cm = BinaryConfusionMatrix(y_valid, y_pred)
                    valid_scores.append(cm.get_metrics(*self.metrics))

                self._validation_history.append(valid_scores)

            if DYNAMIC:
                self.fitting.update()

        for s in solver.pop.get("x"):
            clf = clone(self.estimator)
            clf.fit(X[self.train_ind[s]], y[self.train_ind[s]])
            self._models.append(clf)

        return self

    def get_indicators(self):
        return self._training_history[-1]

    def predict(self, X):
        return np.array([m.predict(X) for m in self._models])

    def _eval_cb(self, X, y):
        def _eval(S):
            scores = []

            for train, X_test, y_test in self.fitting:
                test_scores = []

                for s in S:
                    clf = clone(self.estimator)
                    clf.fit(X[train[s[train]]], y[train[s[train]]])
                    y_pred = clf.predict_proba(X_test)
                    cm = BinaryConfusionMatrix(y_test, y_pred)
                    test_scores.append(cm.get_metrics(*self.metrics))

                scores.append(test_scores)

            scores = np.mean(scores, axis=0)
            return -1 * scores

        return _eval

    def _repair_cb(self, X, y):
        def _repair(s):
            for train, X_test, y_test in self.fitting:
                counts = np.bincount(y[train[s[train]]], minlength=2)

                for missing in np.argwhere(counts == 0):
                    repair_i =  np.random.choice(np.argwhere(y[train] == missing).ravel())
                    s[repair_i] = True

            return s

        return _repair
