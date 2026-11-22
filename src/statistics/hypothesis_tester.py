
import numpy as np
from scipy import stats


class HypothesisTester:
    def __init__(self, alpha: float = 0.05, alternative: str = "two-sided") -> None:
        self.alpha: float = alpha
        self.alternative: str = alternative

    def _test_normality_shapiro(self, differences: np.ndarray) -> tuple[bool, float]:
        if len(differences) < 3 or np.all(differences == differences[0]):
            return False, 1.0
        _stat, p_val = stats.shapiro(differences)
        return bool(p_val >= self.alpha), float(p_val)

    def _compute_effect_size(self, baseline: np.ndarray, filtered: np.ndarray, is_normal: bool) -> float:
        m, n = len(filtered), len(baseline)
        if m < 2 or n < 2:
            return 0.0
        if is_normal:
            var_b, var_f = np.var(baseline, ddof=1), np.var(filtered, ddof=1)
            pooled_std = np.sqrt(((n - 1) * var_b + (m - 1) * var_f) / (n + m - 2))
            return float((np.mean(filtered) - np.mean(baseline)) / pooled_std) if pooled_std > 0 else 0.0
        greater = np.sum(filtered[:, None] > baseline[None, :])
        equal = np.sum(filtered[:, None] == baseline[None, :])
        return float((greater + 0.5 * equal) / (m * n))

    def execute(self, baseline_scores: np.ndarray, filtered_scores: np.ndarray) -> dict[str, float]:
        b = np.asarray(baseline_scores, dtype=np.float64).flatten()
        f = np.asarray(filtered_scores, dtype=np.float64).flatten()
        if len(b) != len(f) or len(b) < 2:
            raise ValueError("Baseline and filtered scores must have identical length >= 2")
        diffs = f - b
        is_normal, shapiro_p = self._test_normality_shapiro(diffs)
        if is_normal:
            p_val = float(stats.ttest_rel(f, b, alternative=self.alternative).pvalue)
        else:
            try:
                p_val = float(stats.wilcoxon(f, b, alternative=self.alternative).pvalue)
            except ValueError:
                p_val = 1.0
        effect_val = self._compute_effect_size(b, f, is_normal)
        return {"p_value": p_val, "is_normal": float(is_normal), "shapiro_p": shapiro_p, "effect_size": effect_val, "significant": float(p_val < self.alpha)}
