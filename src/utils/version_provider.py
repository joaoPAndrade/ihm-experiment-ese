import os
from pathlib import Path


class VersionProvider:
    def __init__(self, version_file: str = "VERSION") -> None:
        self.version_file: Path = Path(version_file)
        self.default_version: str = "0.1.0"

    def _read_from_file(self) -> str:
        with open(self.version_file, "r", encoding="utf-8") as stream:
            return stream.read().strip()

    def execute(self) -> str:
        if self.version_file.is_file():
            return self._read_from_file()
        return self.default_version


if __name__ == "__main__":
    provider = VersionProvider()
    print(provider.execute())
