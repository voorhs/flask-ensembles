import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from math import e
bagging_const = 1 - 1 / e


class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.feature_subsample_size = feature_subsample_size
        self.forest = []
        for i in range(n_estimators):
            self.forest.append(
                DecisionTreeRegressor(max_depth=max_depth, **trees_parameters))

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """
        if self.feature_subsample_size is None:
            self.feature_subsample_size = 1 / 3
        self.subspace_size = int(
            np.ceil(X.shape[1] * self.feature_subsample_size))
        subsample_size = int(np.ceil(X.shape[0] * bagging_const))

        self.subspaces = np.empty(
            (len(self.forest), self.subspace_size), dtype='int')
        rng = np.random.default_rng()
        for i in range(self.n_estimators):
            self.subspaces[i] = rng.choice(X.shape[1],
                                           size=self.subspace_size,
                                           replace=False,
                                           shuffle=False)
            subsample = rng.choice(X.shape[0],
                                   size=subsample_size,
                                   replace=True,
                                   shuffle=False).reshape(-1, 1)
            self.forest[i].fit(X[subsample, self.subspaces[i]], y[subsample])

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        base_predictions = np.empty((self.n_estimators, X.shape[0]))
        for i in range(self.n_estimators):
            base_predictions[i] = self.forest[i].predict(
                X[:, self.subspaces[i]])
        return np.mean(base_predictions, axis=0)


class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use alpha * learning_rate instead of alpha

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.feature_subsample_size = feature_subsample_size
        self.lr = learning_rate
        self.forest = []
        for i in range(n_estimators):
            self.forest.append(
                DecisionTreeRegressor(max_depth=max_depth, **trees_parameters))

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """
        if self.feature_subsample_size is None:
            self.feature_subsample_size = 1 / 3
        self.subspace_size = int(
            np.ceil(X.shape[1] * self.feature_subsample_size))

        # initializing
        self.subspaces = np.empty(
            (self.n_estimators, self.subspace_size), dtype='int')
        self.weights = [1]

        # generate random subspaces
        rng = np.random.default_rng()
        for i in range(self.n_estimators):
            self.subspaces[i] = rng.choice(
                X.shape[1], size=self.subspace_size, replace=False, shuffle=False)

        shift = y.copy()
        predictions = 0
        for i in range(self.n_estimators):
            # fit new tree to approximate shift
            self.forest[i].fit(X[:, self.subspaces[i]], shift)

            # find weight
            # task's loss function is supposed to be here
            # but we handle only MSE
            predicted = self.forest[i].predict(X[:, self.subspaces[i]])
            w = minimize_scalar(
                lambda x: np.sum((x * predicted - shift) ** 2)
            ).x
            self.weights.append(w)
            predictions += self.lr * self.weights[i] * predicted

            shift = y - predictions

        return self

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        ans = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            ans += self.weights[i] * self.lr * \
                self.forest[i].predict(X[:, self.subspaces[i]])
        return ans
