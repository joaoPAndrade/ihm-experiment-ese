
import numpy as np


class DataStandardScaler:
    def __init__(self, epsilon: float = 1e-7) -> None:
        self.epsilon: float = epsilon
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def _compute_train_statistics(self, train_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if train_features.ndim != 2:
            raise ValueError("Training features must be a 2D array")
        mean = np.mean(train_features, axis=0)
        std = np.std(train_features, axis=0)
        std[std == 0.0] = self.epsilon
        return mean, std

    def _apply_zscore_scaling(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler must be fitted before transforming data")
        if data.ndim != 2 or data.shape[1] != self.mean.shape[0]:
            raise ValueError("Input dimension must match fitted feature count")
        return ((data - self.mean) / self.std).astype(np.float32)

    def execute(self, train_features: np.ndarray, test_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.mean, self.std = self._compute_train_statistics(train_features)
        scaled_train = self._apply_zscore_scaling(train_features)
        scaled_test = self._apply_zscore_scaling(test_features)
        return scaled_train, scaled_test
