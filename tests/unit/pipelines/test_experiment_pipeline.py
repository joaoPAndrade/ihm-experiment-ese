import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from src.pipelines.experiment_pipeline import ExperimentPipeline


class TestExperimentPipeline:
    def _test_pipeline_execution(self) -> None:
        pipeline = ExperimentPipeline(random_state=42, n_splits=3, target_size=(32, 32))
        with tempfile.TemporaryDirectory() as temp_dir:
            img_dir = os.path.join(temp_dir, "images")
            os.makedirs(img_dir)
            csv_path = os.path.join(temp_dir, "dataset.csv")
            with open(csv_path, "w", encoding="utf-8") as stream:
                stream.write("filename;specific_epithet\n")
                for i in range(18):
                    species = "SpeciesA" if i < 9 else "SpeciesB"
                    f_name = f"img_{i}.jpg"
                    stream.write(f"{f_name};{species}\n")
                    Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(os.path.join(img_dir, f_name))
            out_dir = os.path.join(temp_dir, "outputs")
            results = pipeline.execute(csv_path, img_dir, out_dir)
            assert "dataset_stats" in results
            assert "q1_spearman_correlations" in results
            assert len(results["q1_spearman_correlations"]) == 13
            assert "q2_hypothesis_test" in results
            assert "plot_paths" in results
            assert len(results["plot_paths"]) == 2
            assert "csv_paths" in results
            assert len(results["csv_paths"]) == 4
            for p in results["csv_paths"]:
                assert os.path.exists(p)

    def _test_pipeline_validation(self) -> None:
        pipeline = ExperimentPipeline()
        with pytest.raises(FileNotFoundError):
            pipeline.execute("non_existent.csv", "/tmp", "/tmp")

    def execute(self) -> None:
        self._test_pipeline_execution()
        self._test_pipeline_validation()

    def test_experiment_pipeline(self) -> None:
        self.execute()
