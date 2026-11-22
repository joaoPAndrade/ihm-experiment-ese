from collections import Counter

import numpy as np
import pytest

from src.validation.stratified_fold_splitter import StratifiedFoldSplitter


class TestStratifiedFoldSplitter:
    def _test_nominal_stratification_and_folds(self) -> None:
        splitter = StratifiedFoldSplitter(n_splits=5, random_state=42)
        labels = np.array([0] * 30 + [1] * 15 + [2] * 10, dtype=np.int64)
        folds = splitter.execute(len(labels), labels)
        assert len(folds) == 5
        seen_test_indices: set[int] = set()
        for train_idx, test_idx in folds:
            assert len(set(train_idx) & set(test_idx)) == 0
            assert len(train_idx) + len(test_idx) == len(labels)
            seen_test_indices.update(test_idx.tolist())
            test_counts = Counter(labels[test_idx].tolist())
            assert test_counts[0] == 6
            assert test_counts[1] == 3
            assert test_counts[2] == 2
        assert seen_test_indices == set(range(len(labels)))

    def _test_validation_errors_and_edge_cases(self) -> None:
        splitter = StratifiedFoldSplitter(n_splits=5)
        with pytest.raises(ValueError):
            splitter.execute(10, np.array([0] * 8))
        with pytest.raises(ValueError):
            splitter.execute(5, np.array([0] * 3 + [1] * 2))
        with pytest.raises(ValueError):
            splitter.execute(0, np.array([]))
        with pytest.raises(ValueError):
            splitter.execute(4, np.zeros((2, 2)))

    def execute(self) -> None:
        self._test_nominal_stratification_and_folds()
        self._test_validation_errors_and_edge_cases()

    def test_stratified_fold_splitter(self) -> None:
        self.execute()
