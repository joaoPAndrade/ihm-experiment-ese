import os
import tempfile
import numpy as np
from PIL import Image
import pytest
from src.data.image_feature_extractor import ImageFeatureExtractor


class TestImageFeatureExtractor:
    def _test_extraction_and_shapes(self) -> None:
        extractor = ImageFeatureExtractor(target_size=(32, 32), n_bins=4, grid_size=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths: list[str] = []
            for idx in range(3):
                path = os.path.join(temp_dir, f"sample_{idx}.png")
                sample_pixels = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
                Image.fromarray(sample_pixels).save(path)
                file_paths.append(path)
            features = extractor.execute(file_paths, batch_size=2)
            assert isinstance(features, np.ndarray)
            assert features.shape[0] == 3
            assert features.shape[1] > 0
            assert not np.isnan(features).any()
            assert not np.isinf(features).any()
            repeat_features = extractor.execute(file_paths, batch_size=2)
            assert np.allclose(features, repeat_features)

    def _test_batching_and_exceptions(self) -> None:
        extractor = ImageFeatureExtractor(target_size=(32, 32))
        with pytest.raises(ValueError):
            extractor.execute([])
        with pytest.raises(FileNotFoundError):
            extractor.execute(["/tmp/non_existent_image_path_123.jpg"])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "test.jpg")
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(path)
            out1 = extractor.execute([path], batch_size=1)
            out2 = extractor.execute([path], batch_size=10)
            assert np.allclose(out1, out2)

    def execute(self) -> None:
        self._test_extraction_and_shapes()
        self._test_batching_and_exceptions()

    def test_image_feature_extractor(self) -> None:
        self.execute()
