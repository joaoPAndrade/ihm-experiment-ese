
import numpy as np


class IhmOutlierFilter:
    def __init__(self, metric_name: str = "kDN", cutoff_percentile: float = 80.0) -> None:
        self.metric_name: str = metric_name
        self.cutoff_percentile: float = cutoff_percentile

    def _determine_keep_mask(self, scores: np.ndarray) -> np.ndarray:
        threshold = float(np.percentile(scores, self.cutoff_percentile))
        keep_mask = scores <= threshold
        if not np.any(keep_mask):
            cutoff = float(np.percentile(scores, 20.0))
            keep_mask = scores <= cutoff
        if not np.any(keep_mask):
            keep_mask[np.argmin(scores)] = True
        return keep_mask

    def _extract_metric_scores(self, ihm_metrics: dict[str, np.ndarray], sample_count: int) -> np.ndarray:
        if self.metric_name in ihm_metrics:
            scores = np.asarray(ihm_metrics[self.metric_name], dtype=np.float64).flatten()
        else:
            all_metric_arrays = [np.asarray(arr, dtype=np.float64).flatten() for arr in ihm_metrics.values()]
            scores = np.mean(all_metric_arrays, axis=0)
        if len(scores) != sample_count:
            raise ValueError(f"Metric array length ({len(scores)}) does not match sample count ({sample_count})")
        return scores

    def execute(self, features: np.ndarray, labels: np.ndarray, ihm_metrics: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(features) != len(labels) or len(features) == 0:
            raise ValueError("Features and labels must have matching non-zero length")
        if not ihm_metrics:
            raise ValueError("IHM metrics dictionary cannot be empty")
        scores = self._extract_metric_scores(ihm_metrics, len(labels))
        keep_mask = self._determine_keep_mask(scores)
        filtered_features = features[keep_mask]
        filtered_labels = labels[keep_mask]
        return filtered_features, filtered_labels, keep_mask
