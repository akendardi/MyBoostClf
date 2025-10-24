import random
from collections import *

import numpy as np
import pandas as pd


class MyBoostClf:

    def __init__(
            self,
            n_estimators: int = 10,
            learning_rate=0.1,
            max_depth: int = 5,
            min_samples_split: int = 2,
            max_leafs: int = 20,
            max_features: float = 0.8,
            max_samples: float = 0.8,
            metric: str = None,
            reg: float = 0.1,
            bins: int = None,
            random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.metric = metric
        self.bins = bins
        self.max_features = max_features
        self.max_samples = max_samples
        self.reg = reg
        self.random_state = random_state

        self.fi = defaultdict(float)
        self.leaves_cnt = 0
        self.pred_0 = 0
        self.trees = []
        self.best_score = 0

    def fit(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            X_eval: pd.DataFrame = None,
            y_eval: pd.Series = None,
            early_stopping: int = None,
            verbose: int = None
    ):
        random.seed(self.random_state)

        self.pred_0 = self.get_log_chance(y)
        old_pred = np.ones(len(y)) * self.pred_0
        y_val = y.values

        if self.metric is None:
            self.best_score = float("inf")
        else:
            self.best_score = -float("inf")

        early_stop_cnt = 0
        best_iter = 0
        for i in range(1, self.n_estimators + 1):
            if verbose is not None and i % verbose == 0:
                log_loss = self.get_log_loss(y, self.get_proba(old_pred))
                log_str = f"{i}. {log_loss}"
                if self.metric is not None:
                    preds = (self.get_proba(old_pred) > 0.5).astype(int)
                    log_str += f" | {self.metric}: {self.get_metric(y, preds, old_pred)}"
                print(f"{i}: {log_loss}")

            proba = self.get_proba(old_pred)
            errors = y_val - proba

            cols_idx = random.sample(range(len(X.columns)), round(self.max_features * len(X.columns)))
            rows_idx = random.sample(range(len(X)), round(self.max_samples * len(X)))

            X_train = X.iloc[rows_idx, cols_idx]
            y_train = pd.Series(errors[rows_idx], index=X_train.index)

            model = MyTreeReg(
                self.max_depth,
                self.min_samples_split,
                self.max_leafs,
                self.bins,
                len(y)
            )

            model.fit(X_train, y_train)
            self.deep_tree(model, y, proba)
            self.leaves_cnt += model.leafs_cnt

            pred = model.predict(X)
            old_pred += self._calculate_learn_rate(i) * pred
            self.trees.append(model)

            if early_stopping is not None:
                eval_pred = self.predict(X_eval)
                eval_pred_proba = self.predict_proba(X_eval)

                score = self.get_score(y_eval, eval_pred, eval_pred_proba)
                if self._is_better(self.best_score, score):
                    self.best_score = score
                    early_stop_cnt = 0
                    best_iter = len(self.trees)
                else:
                    early_stop_cnt += 1

                if early_stop_cnt >= early_stopping:
                    self.trees = self.trees[:best_iter]
                    break

        for feature in X.columns:
            for tree in self.trees:
                self.fi[feature] += tree.fi[feature]

    def _predict_log_odds(self, X: pd.DataFrame):
        f = np.full(len(X), self.pred_0, dtype=float)
        for i in range(1, len(self.trees) + 1):
            tree = self.trees[i - 1]
            f += self._calculate_learn_rate(i) * tree.predict(X)
        return f

    def _calculate_learn_rate(self, iteration: int):
        if callable(self.learning_rate):
            return self.learning_rate(iteration)
        else:
            return self.learning_rate

    def predict_proba(self, X: pd.DataFrame):
        f = self._predict_log_odds(X)
        return self.get_proba(f)

    def predict(self, X: pd.DataFrame):
        preds = self.predict_proba(X)
        return (preds > 0.5).astype(int)

    def deep_tree(self, tree: 'MyTreeReg', y: pd.Series, old_pred: np.array):

        def recursive(node: 'TreeNode'):
            if node.is_leave():
                y_true = y.loc[node.indices].values
                y_pred = old_pred[y.index.get_indexer(node.indices)]
                node.value = self._calculate_learn_rate(len(self.trees) + 1) * self.tailor_poly(y_true, y_pred)
            else:
                recursive(node.left_node)
                recursive(node.right_node)

        recursive(tree.root)

    def tailor_poly(self, y_true, y_pred):
        grad = y_true - y_pred
        hess = y_pred * (1 - y_pred)
        denom = np.sum(hess)
        if denom <= 1e-12 or np.isnan(denom):
            return 0.0
        return np.sum(grad) / denom

    def get_score(self, y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series):
        if self.metric is not None:
            return self.get_metric(y_true, y_pred, y_proba)
        else:
            return self.get_log_loss(y_true, y_proba)

    def _is_better(self, old: float, new: float):
        if self.metric is None:
            return new < old
        else:
            return new > old

    def get_log_loss(self, y_true: pd.Series, proba: np.array):
        eps = 1e-15
        proba = np.clip(proba, eps, 1 - eps)
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        return -np.mean(y_true * np.log(proba) + (1 - y_true) * np.log(1 - proba))

    def get_proba(self, log_odds):
        log_odds = np.clip(log_odds, -700, 700)
        return 1.0 / (1.0 + np.exp(-log_odds))

    def get_log_chance(self, y: pd.Series):
        eps = 1e-15
        if isinstance(y, pd.Series):
            y = y.values
        mean_val = np.clip(np.mean(y), eps, 1 - eps)
        return np.log(mean_val / (1 - mean_val))

    def get_metric(self, y_true: pd.Series, y_pred: pd.Series, y_pred_proba: pd.Series):
        if self.metric == "accuracy":
            return self.get_accuracy(y_true, y_pred)
        if self.metric == "precision":
            return self.get_precision(y_true, y_pred)
        if self.metric == "recall":
            return self.get_recall(y_true, y_pred)
        if self.metric == "f1":
            return self.get_f1(y_true, y_pred)
        if self.metric == "roc_auc":
            return self.get_roc_auc(y_true, y_pred_proba)
        return 0

    def get_accuracy(self, y_true: pd.Series, y_pred: pd.Series):
        tp, fn, fp, tn = self._get_confusion_matrix_np(y_true, y_pred)
        return (tp + tn) / (tp + fn + fp + tn)

    def get_precision(self, y_true: pd.Series, y_pred: pd.Series):
        tp, fn, fp, tn = self._get_confusion_matrix_np(y_true, y_pred)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def get_recall(self, y_true: pd.Series, y_pred: pd.Series):
        tp, fn, fp, tn = self._get_confusion_matrix_np(y_true, y_pred)
        return tp / (tp + fn)

    def get_f1(self, y_true: pd.Series, y_pred: pd.Series):
        recall = self.get_recall(y_true, y_pred)
        precision = self.get_precision(y_true, y_pred)
        return 2 * precision * recall / (precision + recall)

    def get_roc_auc(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        df = pd.DataFrame({"y": y_true, "scores": y_pred})
        df = df.sort_values("scores", ascending=False)

        positives = df[df["y"] == 1]["scores"].values
        negatives = df[df["y"] == 0]["scores"].values

        total = 0
        for neg in negatives:
            score_higher = np.sum(positives > neg)
            score_equal = np.sum(positives == neg)
            total += score_higher + score_equal / 2

        return total / (len(positives) * len(negatives))

    def _get_confusion_matrix_np(self, y, y_pred):
        y = np.array(y)
        y_pred = np.array(y_pred)

        tp = np.sum((y == 1) & (y_pred == 1))
        fn = np.sum((y == 1) & (y_pred == 0))
        fp = np.sum((y == 0) & (y_pred == 1))
        tn = np.sum((y == 0) & (y_pred == 0))

        return tp, fn, fp, tn


class TreeNode:
    def __init__(
            self,
            feature: str = None,
            split_value: float = None,
            left_node: 'TreeNode' = None,
            right_node: 'TreeNode' = None,
            depth: int = 0,
            value: float = None,
            indices: np.array = None
    ):
        self.feature = feature
        self.split_value = split_value
        self.left_node = left_node
        self.right_node = right_node
        self.depth = depth
        self.value = value
        self.indices = indices

    def is_leave(self):
        return self.value is not None


class MyTreeReg:

    def __init__(
            self,
            max_depth: int = 5,
            min_samples_split: int = 2,
            max_leafs: int = 20,
            bins: int = None,
            n: int = 0,

    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.root = None
        self.bins = bins
        self.n = n
        self.histogram = {}
        self.fi = defaultdict(int)
        self.leafs_cnt = 0
        self.sum = 0
        self.features = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self._build_histogram(X)
        self.features = np.array(list(X.columns))
        self.root = self._build_tree(0, X, y, self.max_leafs)

    def _build_histogram(self, X: pd.DataFrame):
        if self.bins is None:
            return
        for feature in X.columns:
            unique_values = np.sort(X[feature].unique())
            if len(unique_values) <= self.bins - 1:
                split_points = (unique_values[1:] + unique_values[:-1]) / 2
                self.histogram[feature] = np.array(split_points)
            else:
                self.histogram[feature] = np.histogram(X[feature].values, bins=self.bins)[1][1:-1]

    def _get_split_points(self, X: pd.DataFrame, feature: str):
        if self.bins is None:
            unique_values = np.sort(X[feature].unique())
            return np.array((unique_values[1:] + unique_values[:-1]) / 2)
        else:
            return self.histogram[feature]

    def _get_feature_importance(self, y, left_y, right_y):
        left_mse = self.get_mse(left_y)
        right_mse = self.get_mse(right_y)
        return len(y) / self.n * (
                self.get_mse(y) - len(left_y) * left_mse / len(y) - len(right_y) * right_mse / len(y))

    def predict(self, X: pd.DataFrame):
        res = []
        for _, row in X[self.features].iterrows():
            curr: TreeNode = self.root
            while not curr.is_leave():
                if row[curr.feature] <= curr.split_value:
                    curr = curr.left_node
                else:
                    curr = curr.right_node
            res.append(curr.value)
        return np.array(res)

    def _build_tree(self, depth: int, X: pd.DataFrame, y: pd.Series, leaves_available: int) -> TreeNode:
        if (depth != 0 and leaves_available <= 1) or depth == self.max_depth or len(
                y) < self.min_samples_split or y.nunique() == 1:
            return self.build_leaf(y, X.index)

        feature, split_point, _ = self.get_best_split(X, y)
        if feature is None:
            return self.build_leaf(y, X.index)

        left_idx = X[X[feature] <= split_point].index
        right_idx = X[X[feature] > split_point].index
        if len(left_idx) == 0 or len(right_idx) == 0:
            return self.build_leaf(y, X.index)

        left_share = max(1, leaves_available - 1)
        left_node = self._build_tree(depth + 1, X.loc[left_idx], y.loc[left_idx], left_share)

        left_used = self.count_leafs(left_node)
        right_share = max(1, leaves_available - left_used)
        right_node = self._build_tree(depth + 1, X.loc[right_idx], y.loc[right_idx], right_share)
        self.fi[feature] += self._get_feature_importance(y, y.loc[left_idx], y.loc[right_idx])

        return TreeNode(
            feature=feature,
            split_value=split_point,
            left_node=left_node,
            right_node=right_node,
            depth=depth,
            indices=X.index
        )

    def count_leafs(self, node: TreeNode = None):
        if node is None:
            node = self.root
        if node.is_leave():
            return 1
        return self.count_leafs(node.left_node) + self.count_leafs(node.right_node)

    def build_leaf(self, y: pd.Series, idx: pd.Index):
        self.leafs_cnt += 1
        val = float(np.mean(y))
        self.sum += val
        return TreeNode(
            value=val,
            indices=idx
        )

    def get_best_split(self, X: pd.DataFrame, y: pd.Series):
        best_res = (None, -1, -1)
        for feature in X.columns:
            sorted_idx = X[feature].sort_values().index
            n = len(y)

            sorted_X: pd.DataFrame = X.loc[sorted_idx]
            sorted_y = y.loc[sorted_idx]

            split_points = self._get_split_points(X, feature)
            for split_point in split_points:
                y1 = sorted_y[sorted_X[feature] <= split_point]
                y2 = sorted_y[sorted_X[feature] > split_point]

                s0 = np.var(y)
                s1 = np.var(y1)
                s2 = np.var(y2)

                s = s0 - len(y1) / n * s1 - len(y2) / n * s2
                if s > best_res[-1]:
                    best_res = (feature, split_point, s)
        return best_res

    def get_mse(self, y: pd.Series):
        n = len(y)
        if n <= 1:
            return 0.0
        y_arr = y.values.astype(float)
        mu = np.mean(y_arr)
        return np.sum((y_arr - mu) ** 2) / n

    def print_tree(self):
        self._recursive_print_tree(self.root, 1)
        print(self.leafs_cnt)
        print(self.sum)

    def _recursive_print_tree(self, root: TreeNode, depth, side: str = ""):
        if root is None:
            return
        if root.is_leave():
            print(f"{' ' * depth}{side} = {root.value}")
        else:
            print(f"{' ' * depth}{root.feature} | {root.split_value}")
            self._recursive_print_tree(root.left_node, depth + 1, "left")
            self._recursive_print_tree(root.right_node, depth + 1, "right")
