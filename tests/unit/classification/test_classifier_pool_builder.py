from src.classification.classifier_pool_builder import ClassifierPoolBuilder


class TestClassifierPoolBuilder:
    def _test_pool_size_and_keys(self) -> None:
        builder = ClassifierPoolBuilder(seed=123)
        pool = builder.execute()
        assert len(pool) == 7
        expected_keys = {
            "SVM_Linear",
            "SVM_RBF",
            "Random_Forest",
            "Gradient_Boosting",
            "Bagging",
            "Logistic_Regression",
            "MLP",
        }
        assert set(pool.keys()) == expected_keys

    def _test_model_seeds(self) -> None:
        builder = ClassifierPoolBuilder(seed=999)
        pool = builder.execute()
        for model in pool.values():
            assert getattr(model, "random_state", None) == 999

    def execute(self) -> None:
        self._test_pool_size_and_keys()
        self._test_model_seeds()

    def test_classifier_pool_builder(self) -> None:
        self.execute()
