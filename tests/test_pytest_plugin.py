from __future__ import annotations

import json
import textwrap
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from llm_cassette.pytest_plugin import _safe_component

pytest_plugins = ["pytester"]

_HEADER = """
from pathlib import Path

from llm_cassette import current_cassette, tool

_CALLS = Path(__file__).resolve().with_name("calls.txt")


@tool
def greet(name: str) -> dict[str, object]:
    count = int(_CALLS.read_text(encoding="utf-8")) if _CALLS.exists() else 0
    _CALLS.write_text(str(count + 1), encoding="utf-8")
    return {"name": name, "n": count + 1}
"""


def test_pytest11_entry_point() -> None:
    matches = [
        entry
        for entry in entry_points(group="pytest11")
        if entry.value == "llm_cassette.pytest_plugin"
    ]
    assert [entry.name for entry in matches] == ["llm_cassette"]


def test_safe_component_keeps_ids_distinct_and_portable() -> None:
    samples = [
        "test_city[Paris]",
        "test_city[New York]",
        "test_city[a/b]",
        "test_city[a_b]",
        "test_city[a%2Fb]",
        "AUX",
        "AUX%5F",
        "aux",
        "CON.txt",
        "..",
        ".",
        "",
        "a\\b",
        "a:b",
        "東京",
        "a" * 300,
        ("a" * 300) + "b",
    ]
    encoded = [_safe_component(sample) for sample in samples]
    assert len(set(encoded)) == len(samples)
    for name in encoded:
        assert "/" not in name
        assert "\\" not in name
        assert ":" not in name
        assert name not in {"", ".", ".."}
        assert name.split(".", 1)[0].upper() not in {"CON", "PRN", "AUX", "NUL"}
        assert len(name.encode("utf-8")) <= 180
    assert _safe_component("test_city[Paris]") == "test_city[Paris]"
    assert _safe_component("test_city[a/b]") == "test_city[a%2Fb]"
    assert _safe_component("test_city[a%2Fb]") == "test_city[a%252Fb]"
    assert _safe_component("AUX") == "AUX%5F"
    assert _safe_component("CON.txt") == "CON%5F.txt"


def test_auto_names_are_next_to_the_test_module(pytester: pytest.Pytester) -> None:
    _write(
        pytester,
        "tests/test_agent.py",
        """
        import pytest

        @pytest.mark.cassette
        def test_marked() -> None:
            assert current_cassette() is not None
            assert greet("Ada")["name"] == "Ada"

        def test_fixture(cassette: object) -> None:
            assert current_cassette() is cassette
            root = Path(__file__).parent / "cassettes" / "test_agent"
            assert getattr(cassette, "path") == root / "test_fixture.jsonl"
            assert greet("Grace")["name"] == "Grace"
        """,
    )
    _write(
        pytester,
        "tests/agents/test_chat.py",
        """
        import pytest

        @pytest.mark.cassette
        def test_hi() -> None:
            assert greet("Lin")["name"] == "Lin"
        """,
    )

    first = pytester.runpytest("tests")
    first.assert_outcomes(passed=3)
    marked = _cassette(pytester, "tests/test_agent.py", "test_marked.jsonl")
    fixture = _cassette(pytester, "tests/test_agent.py", "test_fixture.jsonl")
    nested = _cassette(pytester, "tests/agents/test_chat.py", "test_hi.jsonl")
    assert json.loads(marked.read_text(encoding="utf-8").splitlines()[1])["args"] == {
        "name": "Ada"
    }
    assert _calls(pytester, "tests/test_agent.py") == 2
    assert _calls(pytester, "tests/agents/test_chat.py") == 1

    second = pytester.runpytest("tests")
    second.assert_outcomes(passed=3)
    assert _calls(pytester, "tests/test_agent.py") == 2
    assert _calls(pytester, "tests/agents/test_chat.py") == 1
    assert fixture.is_file()
    assert nested.is_file()


def test_unmarked_test_writes_nothing(pytester: pytest.Pytester) -> None:
    _write(
        pytester,
        "tests/test_plain.py",
        """
        def test_plain() -> None:
            assert True
        """,
        tools=False,
    )
    result = pytester.runpytest("tests")
    result.assert_outcomes(passed=1)
    assert not (pytester.path / "tests" / "cassettes").exists()


def test_replay_mode_option(pytester: pytest.Pytester) -> None:
    _write(
        pytester,
        "tests/test_agent.py",
        """
        def test_greet(cassette: object) -> None:
            assert greet("Ada")["name"] == "Ada"
        """,
    )
    recorded = pytester.runpytest("tests")
    recorded.assert_outcomes(passed=1)
    cassette = _cassette(pytester, "tests/test_agent.py", "test_greet.jsonl")
    assert cassette.is_file()
    assert _calls(pytester, "tests/test_agent.py") == 1

    replay = pytester.runpytest("tests", "--replay-mode=replay")
    replay.assert_outcomes(passed=1)
    assert _calls(pytester, "tests/test_agent.py") == 1

    cassette.unlink()
    missing = pytester.runpytest("tests", "--replay-mode=replay")
    missing.assert_outcomes(errors=1)
    output = _output(missing)
    assert "CassetteNotFoundError" in output
    assert "cassette not found" in output
    assert _calls(pytester, "tests/test_agent.py") == 1

    passthrough = pytester.runpytest("tests", "--replay-mode=none")
    passthrough.assert_outcomes(passed=1)
    assert not cassette.exists()
    assert _calls(pytester, "tests/test_agent.py") == 2

    record = pytester.runpytest("tests", "--replay-mode=record")
    record.assert_outcomes(passed=1)
    assert _calls(pytester, "tests/test_agent.py") == 3
    event = json.loads(cassette.read_text(encoding="utf-8").splitlines()[1])
    assert event["result"] == {"n": 3, "name": "Ada"}

    again = pytester.runpytest("tests", "--replay-mode=record")
    again.assert_outcomes(passed=1)
    assert _calls(pytester, "tests/test_agent.py") == 4

    rejected = pytester.runpytest("tests", "--replay-mode=sideways")
    assert rejected.ret != 0
    assert "sideways" in _output(rejected)

    help_result = pytester.runpytest("--help")
    assert help_result.ret == 0
    help_result.stdout.fnmatch_lines(["*--replay-mode*"])


def test_pyproject_replay_mode(pytester: pytest.Pytester) -> None:
    _write(
        pytester,
        "tests/test_agent.py",
        """
        import pytest

        @pytest.mark.cassette
        def test_greet() -> None:
            assert greet("Ada")["name"] == "Ada"
        """,
    )
    pytester.makepyprojecttoml(
        """
        [tool.pytest.ini_options]
        replay_mode = "replay"
        """
    )
    missing = pytester.runpytest("tests")
    missing.assert_outcomes(errors=1)
    assert "CassetteNotFoundError" in _output(missing)

    recorded = pytester.runpytest("tests", "--replay-mode=once")
    recorded.assert_outcomes(passed=1)
    assert _cassette(pytester, "tests/test_agent.py", "test_greet.jsonl").is_file()
    assert _calls(pytester, "tests/test_agent.py") == 1

    replayed = pytester.runpytest("tests")
    replayed.assert_outcomes(passed=1)
    assert _calls(pytester, "tests/test_agent.py") == 1

    pytester.makepyprojecttoml(
        """
        [tool.pytest.ini_options]
        replay_mode = "sideways"
        """
    )
    rejected = pytester.runpytest("tests", "--replay-mode=once")
    assert rejected.ret != 0
    assert "unknown replay_mode" in _output(rejected)
    assert "sideways" in _output(rejected)


def test_tox_ini_replay_mode(pytester: pytest.Pytester) -> None:
    _write(
        pytester,
        "tests/test_agent.py",
        """
        import pytest

        @pytest.mark.cassette
        def test_greet() -> None:
            greet("Ada")
        """,
    )
    pytester.makeini(
        """
        [pytest]
        replay_mode = replay
        """
    )
    missing = pytester.runpytest("tests")
    missing.assert_outcomes(errors=1)
    assert "cassette not found" in _output(missing)

    recorded = pytester.runpytest("tests", "--replay-mode=record")
    recorded.assert_outcomes(passed=1)
    assert _cassette(pytester, "tests/test_agent.py", "test_greet.jsonl").is_file()


def test_marker_name_and_mode_override(pytester: pytest.Pytester) -> None:
    _write(
        pytester,
        "tests/test_agent.py",
        """
        import pytest

        @pytest.mark.cassette("weather", mode="record")
        def test_named(cassette: object) -> None:
            assert current_cassette() is cassette
            assert getattr(cassette, "mode") == "record"
            assert getattr(cassette, "path").name == "weather.jsonl"
            assert greet("Ada")["name"] == "Ada"

        @pytest.mark.cassette(name="kept.jsonl", mode="replay")
        def test_replay_only() -> None:
            greet("Ada")

        @pytest.mark.parametrize("city", ["Paris", "Lyon"])
        @pytest.mark.cassette("cities", mode="once")
        def test_cities(city: str) -> None:
            assert greet(city)["name"] == city
        """,
    )
    result = pytester.runpytest("tests", "--replay-mode=none")
    result.assert_outcomes(passed=3, errors=1)
    output = _output(result)
    assert "CassetteNotFoundError" in output
    assert "test_replay_only" in output
    weather = _cassette(pytester, "tests/test_agent.py", "weather.jsonl")
    assert weather.is_file()
    assert not _cassette(pytester, "tests/test_agent.py", "test_named.jsonl").exists()
    paris = _cassette(pytester, "tests/test_agent.py", "cities[Paris].jsonl")
    lyon = _cassette(pytester, "tests/test_agent.py", "cities[Lyon].jsonl")
    assert paris.is_file()
    assert lyon.is_file()
    assert json.loads(weather.read_text(encoding="utf-8").splitlines()[1])["args"] == {
        "name": "Ada"
    }
    assert _calls(pytester, "tests/test_agent.py") == 3

    again = pytester.runpytest("tests", "--replay-mode=none")
    again.assert_outcomes(passed=3, errors=1)
    assert _calls(pytester, "tests/test_agent.py") == 4
    event = json.loads(weather.read_text(encoding="utf-8").splitlines()[1])
    assert event["result"] == {
        "n": _calls(pytester, "tests/test_agent.py"),
        "name": "Ada",
    }


def test_marker_rejects_a_bad_name_or_mode(pytester: pytest.Pytester) -> None:
    _write(
        pytester,
        "tests/test_agent.py",
        """
        import pytest

        @pytest.mark.cassette("weather", name="other")
        def test_both() -> None:
            greet("Ada")

        @pytest.mark.cassette(mode="sideways")
        def test_mode() -> None:
            greet("Ada")
        """,
    )
    result = pytester.runpytest("tests")
    result.assert_outcomes(errors=2)
    output = _output(result)
    assert "positional argument or as name=" in output
    assert "unknown cassette mode 'sideways'" in output
    assert not (pytester.path / "tests" / "cassettes").exists()
    assert _calls(pytester, "tests/test_agent.py") == 0


def test_mismatch_is_a_test_failure(pytester: pytest.Pytester) -> None:
    path = _write(
        pytester,
        "tests/test_agent.py",
        """
        import pytest

        @pytest.mark.cassette
        def test_greet() -> None:
            assert greet("Ada")["name"] == "Ada"
        """,
    )
    recorded = pytester.runpytest("tests")
    recorded.assert_outcomes(passed=1)

    path.write_text(
        path.read_text(encoding="utf-8").replace('greet("Ada")', 'greet("Grace")'),
        encoding="utf-8",
    )
    failed = pytester.runpytest("tests", "--replay-mode=replay")
    failed.assert_outcomes(failed=1)
    output = _output(failed)
    assert "CassetteMismatchError" in output
    assert "request does not match" in output
    assert "expected:" in output
    assert "received:" in output
    assert "Ada" in output
    assert "Grace" in output
    assert _calls(pytester, "tests/test_agent.py") == 1


def test_parametrized_ids_get_unique_safe_filenames(pytester: pytest.Pytester) -> None:
    _write(
        pytester,
        "tests/test_agent.py",
        """
        import pytest

        @pytest.mark.cassette
        @pytest.mark.parametrize("city", ["Paris", "New York", "a/b", "a_b", "a%2Fb"])
        def test_city(city: str) -> None:
            assert greet(city)["name"] == city

        @pytest.mark.cassette
        class TestLeft:
            def test_city(self) -> None:
                assert greet("Paris")["name"] == "Paris"

        class TestRight:
            @pytest.mark.cassette
            def test_city(self, cassette: object) -> None:
                assert getattr(cassette, "path").name == "test_city.jsonl"
                assert greet("Lyon")["name"] == "Lyon"
        """,
    )
    first = pytester.runpytest("tests")
    first.assert_outcomes(passed=7)
    directory = pytester.path / "tests" / "cassettes" / "test_agent"
    expected = [
        directory / "test_city[Paris].jsonl",
        directory / "test_city[New%20York].jsonl",
        directory / "test_city[a%2Fb].jsonl",
        directory / "test_city[a_b].jsonl",
        directory / "test_city[a%252Fb].jsonl",
        directory / "TestLeft" / "test_city.jsonl",
        directory / "TestRight" / "test_city.jsonl",
    ]
    for path in expected:
        assert path.is_file()
    assert len({path.resolve() for path in expected}) == len(expected)
    assert not (directory / "test_city[a").exists()
    assert _calls(pytester, "tests/test_agent.py") == 7

    second = pytester.runpytest("tests", "--replay-mode=replay")
    second.assert_outcomes(passed=7)
    assert _calls(pytester, "tests/test_agent.py") == 7


def _write(
    pytester: pytest.Pytester,
    relative: str,
    source: str,
    *,
    tools: bool = True,
) -> Path:
    body = textwrap.dedent(source).strip() + "\n"
    if tools:
        body = f"{_HEADER.strip()}\n\n{body}"
    path = pytester.path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _cassette(pytester: pytest.Pytester, relative: str, name: str) -> Path:
    module = Path(relative)
    return pytester.path / module.parent / "cassettes" / module.stem / name


def _calls(pytester: pytest.Pytester, relative: str) -> int:
    path = (pytester.path / relative).with_name("calls.txt")
    if not path.is_file():
        return 0
    return int(path.read_text(encoding="utf-8"))


def _output(result: pytest.RunResult) -> str:
    return f"{result.stdout}\n{result.stderr}"
