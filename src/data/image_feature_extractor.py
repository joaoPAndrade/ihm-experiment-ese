import os

import numpy as np
from PIL import Image, ImageFile


class ImageFeatureExtractor:
    def __init__(self, target_size: tuple[int, int] = (64, 64), n_bins: int = 8, grid_size: int = 2) -> None:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.target_size: tuple[int, int] = target_size
        self.n_bins: int = n_bins
        self.grid_size: int = grid_size

    def _extract_single_features(self, image_path: str) -> np.ndarray:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        with Image.open(image_path) as raw_img:
            image_array = np.asarray(raw_img.convert("RGB").resize(self.target_size), dtype=np.float32) / 255.0
        feature_parts: list[np.ndarray] = []
        for channel in range(3):
            data = image_array[:, :, channel]
            mean_val = float(np.mean(data))
            std_val = float(np.std(data)) + 1e-7
            skew_val = float(np.mean(((data - mean_val) / std_val) ** 3))
            hist, _ = np.histogram(data, bins=self.n_bins, range=(0.0, 1.0), density=True)
            feature_parts.extend([np.array([mean_val, std_val, skew_val], dtype=np.float32), hist.astype(np.float32)])
        gray = np.mean(image_array, axis=-1)
        grad_x = np.diff(gray, axis=1)
        grad_y = np.diff(gray, axis=0)
        grad_stats = np.array([np.mean(np.abs(grad_x)), np.std(grad_x), np.mean(np.abs(grad_y)), np.std(grad_y)], dtype=np.float32)
        feature_parts.append(grad_stats)
        gh, gw = self.target_size[0] // self.grid_size, self.target_size[1] // self.grid_size
        grid_stats: list[float] = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                cell = image_array[row * gh : (row + 1) * gh, col * gw : (col + 1) * gw, :]
                grid_stats.extend([float(np.mean(cell)), float(np.std(cell))])
        feature_parts.append(np.array(grid_stats, dtype=np.float32))
        return np.concatenate(feature_parts)

    def _process_batch_chunk(self, batch_paths: list[str]) -> np.ndarray:
        batch_vectors = [self._extract_single_features(path) for path in batch_paths]
        return np.vstack(batch_vectors)

    def execute(self, image_paths: list[str], batch_size: int = 32) -> np.ndarray:
        if not image_paths:
            raise ValueError("The image_paths parameter cannot be empty")
        batch_outputs: list[np.ndarray] = []
        for start_idx in range(0, len(image_paths), max(1, batch_size)):
            chunk = image_paths[start_idx : start_idx + batch_size]
            batch_outputs.append(self._process_batch_chunk(chunk))
        final_matrix = np.vstack(batch_outputs)
        return np.nan_to_num(final_matrix, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
