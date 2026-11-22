import os
import tempfile
from typing import Tuple
import numpy as np
import pytest
from src.data.botanical_dataset_loader import BotanicalDatasetLoader


class TestBotanicalDatasetLoader:
    def _test_nominal_loading_and_filtering(self) -> None:
        loader = BotanicalDatasetLoader(min_samples_per_class=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            images_dir = os.path.join(temp_dir, "images")
            os.makedirs(images_dir)
            for file_name in ("img1.jpg", "img2.jpg", "img3.jpg", "img4"):
                open(os.path.join(images_dir, file_name), "w", encoding="utf-8").close()
            csv_path = os.path.join(temp_dir, "data.csv")
            with open(csv_path, "w", encoding="utf-8") as stream:
                stream.write("filename;specific_epithet;herbaria\n")
                stream.write("img1;Piper nigrum;ALCB\n")
                stream.write("img2;Piper nigrum;ALCB\n")
                stream.write("img3;Peperomia blanda;VIES\n")
                stream.write("img4;Piper aduncum;VIES\n")
            paths, labels, mapping = loader.execute(csv_path, images_dir)
            assert len(paths) == 2
            assert len(labels) == 2
            assert mapping == {"Piper nigrum": 0}
            assert np.all(labels == 0)

    def _test_invalid_paths_and_exceptions(self) -> None:
        loader = BotanicalDatasetLoader()
        with pytest.raises(FileNotFoundError):
            loader.execute("non_existent.csv", "/tmp")
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_csv = os.path.join(temp_dir, "invalid.csv")
            with open(invalid_csv, "w", encoding="utf-8") as stream:
                stream.write("col_a;col_b\n1;2\n")
            with pytest.raises(ValueError):
                loader.execute(invalid_csv, temp_dir)
            valid_csv = os.path.join(temp_dir, "valid.csv")
            with open(valid_csv, "w", encoding="utf-8") as stream:
                stream.write("filename;specific_epithet\nimg1;sp1\n")
            with pytest.raises(FileNotFoundError):
                loader.execute(valid_csv, os.path.join(temp_dir, "non_existent_dir"))

    def execute(self) -> None:
        self._test_nominal_loading_and_filtering()
        self._test_invalid_paths_and_exceptions()

    def test_botanical_dataset_loader(self) -> None:
        self.execute()
