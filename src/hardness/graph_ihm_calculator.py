import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
from scipy.spatial.distance import pdist, squareform


class GraphIhmCalculator:
    def __init__(self, distance_percentile: float = 15.0) -> None:
        self.distance_percentile: float = distance_percentile

    def _compute_local_set_metrics(self, dist_mat: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
        n = len(labels)
        max_dist = max(1e-7, float(np.max(dist_mat)))
        lsc, lsr = np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)
        u_counts, h_counts = np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)
        for i in range(n):
            same_mask, diff_mask = labels == labels[i], labels != labels[i]
            if np.any(diff_mask):
                diff_indices = np.where(diff_mask)[0]
                nearest_enemy_local = int(np.argmin(dist_mat[i, diff_indices]))
                r_enemy = float(dist_mat[i, diff_indices[nearest_enemy_local]])
                h_counts[diff_indices[nearest_enemy_local]] += 1.0
            else:
                r_enemy = max_dist
            ls_mask = (dist_mat[i] < r_enemy) & same_mask
            u_counts[ls_mask] += 1.0
            lsc[i] = 1.0 - (float(np.sum(ls_mask)) / max(1, int(np.sum(same_mask))))
            lsr[i] = 1.0 - min(1.0, r_enemy / max_dist)
        u_vals = 1.0 - np.clip(u_counts / max(1, n), 0.0, 1.0)
        h_vals = np.clip(h_counts / max(1, n), 0.0, 1.0)
        return {"LSC_IHM": np.clip(lsc, 0.0, 1.0), "LSR_IHM": np.clip(lsr, 0.0, 1.0), "U": u_vals, "H": h_vals}

    def _compute_mst_and_graph_metrics(self, dist_mat: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
        n = len(labels)
        dist_inf = dist_mat.copy()
        np.fill_diagonal(dist_inf, np.inf)
        n2 = np.zeros(n, dtype=np.float64)
        for i in range(n):
            same_d = dist_inf[i, labels == labels[i]]
            d_intra = float(np.min(same_d)) if np.any(np.isfinite(same_d)) else 0.0
            diff_d = dist_inf[i, labels != labels[i]]
            d_inter = float(np.min(diff_d)) if np.any(np.isfinite(diff_d)) else d_intra + 1.0
            n2[i] = d_intra / (d_intra + d_inter) if (d_intra + d_inter) > 0 else 0.0
        mst = minimum_spanning_tree(dist_mat).toarray()
        mst_sym = np.maximum(mst, mst.T) > 0
        n1 = np.array([float(np.any(labels[mst_sym[i]] != labels[i])) if np.any(mst_sym[i]) else 0.0 for i in range(n)])
        upper = dist_mat[np.triu_indices(n, k=1)]
        cutoff = float(np.percentile(upper, self.distance_percentile)) if len(upper) > 0 else 0.0
        adj = ((dist_mat <= cutoff) & (dist_mat > 0)).astype(np.float64)
        same_adj = adj * (labels[:, None] == labels[None, :])
        deg = np.array([1.0 - np.sum(same_adj[i]) / max(1, int(np.sum(labels == labels[i])) - 1) for i in range(n)])
        sp = shortest_path(same_adj, directed=False, unweighted=True)
        close = np.array([1.0 - np.sum([1.0 / d for d in sp[i, (labels == labels[i]) & (np.arange(n) != i)] if np.isfinite(d) and d > 0]) / max(1, int(np.sum(labels == labels[i])) - 1) for i in range(n)])
        return {"N1_IHM": np.clip(n1, 0.0, 1.0), "N2_IHM": np.clip(n2, 0.0, 1.0), "Degree_IHM": np.clip(deg, 0.0, 1.0), "Closeness_IHM": np.clip(close, 0.0, 1.0)}

    def execute(self, features: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
        if len(features) != len(labels) or len(labels) < 2:
            raise ValueError("Features and labels must have matching length >= 2")
        dist_mat = squareform(pdist(features))
        metrics = self._compute_local_set_metrics(dist_mat, labels)
        metrics.update(self._compute_mst_and_graph_metrics(dist_mat, labels))
        return metrics
