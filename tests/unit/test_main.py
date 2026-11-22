import os
import tempfile

import numpy as np
from PIL import Image

from main import Main


class TestMain:
    def _test_main_cli_execution(self) -> None:
        runner = Main()
        with tempfile.TemporaryDirectory() as temp_dir:
            img_dir = os.path.join(temp_dir, "images")
            os.makedirs(img_dir)
            csv_path = os.path.join(temp_dir, "dataset.csv")
            with open(csv_path, "w", encoding="utf-8") as stream:
                stream.write("filename;specific_epithet\n")
                for i in range(15):
                    species = "SpeciesA" if i < 8 else "SpeciesB"
                    f_name = f"img_{i}.jpg"
                    stream.write(f"{f_name};{species}\n")
                    Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(os.path.join(img_dir, f_name))
            out_dir = os.path.join(temp_dir, "outputs")
            args = ["--csv-path", csv_path, "--images-dir", img_dir, "--output-dir", out_dir, "--n-splits", "3", "--max-samples", "15"]
            status = runner.execute(args)
            assert status == 0
            assert os.path.exists(os.path.join(out_dir, "instance_hardness_and_ihm.csv"))
            assert os.path.exists(os.path.join(out_dir, "q1_spearman_correlations.csv"))

    def execute(self) -> None:
        self._test_main_cli_execution()

    def test_main_execution(self) -> None:
        self.execute()
