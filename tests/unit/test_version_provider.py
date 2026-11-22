from src.utils.version_provider import VersionProvider


class TestVersionProvider:
    def _test_default_version(self) -> None:
        provider = VersionProvider(version_file="NON_EXISTENT_VERSION_FILE")
        assert provider.execute() == "0.1.0"

    def _test_existing_version_file(self) -> None:
        provider = VersionProvider(version_file="VERSION")
        assert provider.execute() == "0.1.0"

    def execute(self) -> None:
        self._test_default_version()
        self._test_existing_version_file()

    def test_version_provider(self) -> None:
        self.execute()
