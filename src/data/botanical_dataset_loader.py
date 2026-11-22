import csv
import os
from collections import Counter

import numpy as np


class BotanicalDatasetLoader:
    def __init__(self, min_samples_per_class: int = 5) -> None:
        self.min_samples_per_class: int = min_samples_per_class

    def _parse_metadata_csv(self, csv_path: str) -> list[tuple[str, str]]:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")
        with open(csv_path, mode="r", encoding="utf-8", errors="ignore") as stream:
            sample_header = stream.readline()
            delimiter = ";" if ";" in sample_header else ","
            stream.seek(0)
            reader = csv.DictReader(stream, delimiter=delimiter)
            if not reader.fieldnames or "filename" not in reader.fieldnames or "specific_epithet" not in reader.fieldnames:
                raise ValueError("CSV must contain 'filename' and 'specific_epithet' columns")
            records = [(row["filename"].strip(), row["specific_epithet"].strip()) for row in reader if row.get("filename") and row.get("specific_epithet")]
        return records

    def _resolve_and_filter_dataset(self, records: list[tuple[str, str]], images_dir: str) -> tuple[list[str], list[str]]:
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Image directory not found: {images_dir}")
        resolved_pairs: list[tuple[str, str]] = []
        for filename, species in records:
            direct_path = os.path.join(images_dir, filename)
            for candidate in (direct_path, f"{direct_path}.jpg", f"{direct_path}.jpeg", f"{direct_path}.png"):
                if os.path.isfile(candidate):
                    resolved_pairs.append((candidate, species))
                    break
        species_counts = Counter(species for _, species in resolved_pairs)
        valid_pairs = [(path, species) for path, species in resolved_pairs if species_counts[species] >= self.min_samples_per_class]
        image_paths = [path for path, _ in valid_pairs]
        species_labels = [species for _, species in valid_pairs]
        return image_paths, species_labels

    def execute(self, csv_path: str, images_dir: str) -> tuple[list[str], np.ndarray, dict[str, int]]:
        raw_records = self._parse_metadata_csv(csv_path)
        valid_paths, valid_labels = self._resolve_and_filter_dataset(raw_records, images_dir)
        unique_species = sorted(set(valid_labels))
        label_to_index = {species: index for index, species in enumerate(unique_species)}
        encoded_labels = np.array([label_to_index[species] for species in valid_labels], dtype=np.int64)
        return valid_paths, encoded_labels, label_to_index
