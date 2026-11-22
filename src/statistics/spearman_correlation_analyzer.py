
import numpy as np
from scipy import stats


class SpearmanCorrelationAnalyzer:
    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha: float = alpha

    def _compute_pair_spearman(self, ground_truth: np.ndarray, metric_values: np.ndarray) -> tuple[float, float, bool]:
        valid_mask = np.isfinite(ground_truth) & np.isfinite(metric_values)
        if np.sum(valid_mask) < 3:
            return 0.0, 1.0, False
        sub_gt, sub_metric = ground_truth[valid_mask], metric_values[valid_mask]
        if np.all(sub_gt == sub_gt[0]) or np.all(sub_metric == sub_metric[0]):
            return 0.0, 1.0, False
        res = stats.spearmanr(sub_gt, sub_metric)
        rho = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
        p_val = float(res.pvalue) if hasattr(res, "pvalue") else float(res[1])
        if not np.isfinite(rho):
            rho = 0.0
        return rho, p_val, bool(p_val < self.alpha)

    def _rank_correlations(self, correlation_map: dict[str, tuple[float, float, bool]]) -> dict[str, tuple[float, float, bool]]:
        sorted_items = sorted(correlation_map.items(), key=lambda item: abs(item[1][0]), reverse=True)
        return dict(sorted_items)

    def execute(self, ih_scores: np.ndarray, ihm_metrics: dict[str, np.ndarray]) -> dict[str, tuple[float, float, bool]]:
        flat_ih = np.asarray(ih_scores, dtype=np.float64).flatten()
        if not ihm_metrics:
            raise ValueError("IHM metrics dictionary cannot be empty")
        results: dict[str, tuple[float, float, bool]] = {}
        for name, values in ihm_metrics.items():
            flat_vals = np.asarray(values, dtype=np.float64).flatten()
            if len(flat_ih) != len(flat_vals):
                raise ValueError(f"Length mismatch: IH has {len(flat_ih)}, metric '{name}' has {len(flat_vals)}")
            results[name] = self._compute_pair_spearman(flat_ih, flat_vals)
        return self._rank_correlations(results)
