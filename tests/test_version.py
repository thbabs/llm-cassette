from importlib.metadata import version

from llm_cassette import __version__


def test_version() -> None:
    assert __version__ == version("llm-cassette") == "0.0.1"
