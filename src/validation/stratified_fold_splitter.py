from collections import Counter
from typing import List, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold


class StratifiedFoldSplitter:
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42) -> None:
        self.n_splits: int = n_splits
        self.shuffle: bool = shuffle
        self.random_state: int = random_state

    def _validate_label_counts(self, labels: np.ndarray) -> None:
        if labels.ndim != 1 or len(labels) == 0:
            raise ValueError("Labels array must be 1D and non-empty")
        class_counts = Counter(labels.tolist())
        min_count = min(class_counts.values())
        if min_count < self.n_splits:
            raise ValueError(f"Minority class has only {min_count} samples, minimum required is {self.n_splits}")

    def _generate_stratified_indices(self, labels: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        dummy_features = np.zeros((len(labels), 1), dtype=np.float32)
        folds_list: List[Tuple[np.ndarray, np.ndarray]] = []
        for train_idx, test_idx in skf.split(dummy_features, labels):
            folds_list.append((train_idx.astype(np.int64), test_idx.astype(np.int64)))
        return folds_list

    def execute(self, feature_count: int, labels: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        if feature_count != len(labels):
            raise ValueError(f"Feature count ({feature_count}) does not match labels length ({len(labels)})")
        self._validate_label_counts(labels)
        return self._generate_stratified_indices(labels)
