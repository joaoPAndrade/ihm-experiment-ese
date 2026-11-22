import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier

from src.hardness.graph_ihm_calculator import GraphIhmCalculator


class IhmMetricsExtractor:
    def __init__(self, k_neighbors: int = 5, seed: int = 42) -> None:
        self.k_neighbors: int = k_neighbors
        self.seed: int = seed

    def _compute_neighborhood_and_overlap_metrics(self, features: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
        n, d = features.shape
        k_val = max(1, min(self.k_neighbors, n - 1))
        nn = NearestNeighbors(n_neighbors=k_val + 1).fit(features)
        _, indices = nn.kneighbors(features)
        kdn = np.array([np.mean(labels[indices[i, 1:]] != labels[i]) for i in range(n)], dtype=np.float64)
        f1_ihm = np.zeros(n, dtype=np.float64)
        unique_classes = np.unique(labels)
        if len(unique_classes) > 1:
            overlap_counts = np.zeros(n, dtype=np.float64)
            for j in range(d):
                feat_col = features[:, j]
                for cls in unique_classes:
                    other_vals = feat_col[labels != cls]
                    own_idx = np.where(labels == cls)[0]
                    in_overlap = (feat_col[own_idx] >= np.min(other_vals)) & (feat_col[own_idx] <= np.max(other_vals))
                    overlap_counts[own_idx] += in_overlap.astype(np.float64)
            f1_ihm = overlap_counts / max(1, d)
        return {"kDN": np.clip(kdn, 0.0, 1.0), "F1_IHM": np.clip(f1_ihm, 0.0, 1.0)}

    def _compute_tree_and_likelihood_metrics(self, features: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
        n = len(labels)
        dt = DecisionTreeClassifier(min_samples_leaf=max(2, min(5, n // 10)), random_state=self.seed).fit(features, labels)
        leaf_nodes, tree_obj = dt.apply(features), dt.tree_
        node_depths = np.zeros(tree_obj.node_count, dtype=np.float64)
        stack = [(0, 0)]
        while stack:
            n_id, depth = stack.pop()
            node_depths[n_id] = depth
            if tree_obj.children_left[n_id] != -1:
                stack.append((int(tree_obj.children_left[n_id]), depth + 1))
            if tree_obj.children_right[n_id] != -1:
                stack.append((int(tree_obj.children_right[n_id]), depth + 1))
        td = np.array([node_depths[leaf_nodes[i]] / max(1.0, float(np.max(node_depths))) for i in range(n)], dtype=np.float64)
        dcp = np.zeros(n, dtype=np.float64)
        for leaf in np.unique(leaf_nodes):
            in_leaf = np.where(leaf_nodes == leaf)[0]
            dcp[in_leaf] = [np.mean(labels[in_leaf] != labels[idx]) for idx in in_leaf]
        gnb = GaussianNB().fit(features, labels)
        probs, cls_map = gnb.predict_proba(features), {cls: idx for idx, cls in enumerate(gnb.classes_)}
        cld = np.zeros(n, dtype=np.float64)
        for i in range(n):
            p_own = probs[i, cls_map[labels[i]]]
            rival = max([probs[i, c] for c in range(probs.shape[1]) if c != cls_map[labels[i]]], default=0.0)
            cld[i] = (1.0 - (p_own - rival)) / 2.0
        return {"TD": np.clip(td, 0.0, 1.0), "DCP": np.clip(dcp, 0.0, 1.0), "CLD": np.clip(cld, 0.0, 1.0)}

    def execute(self, features: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
        if len(features) != len(labels) or len(labels) < 2:
            raise ValueError("Features and labels must have matching length >= 2")
        metrics = self._compute_neighborhood_and_overlap_metrics(features, labels)
        metrics.update(self._compute_tree_and_likelihood_metrics(features, labels))
        metrics.update(GraphIhmCalculator().execute(features, labels))
        return metrics
