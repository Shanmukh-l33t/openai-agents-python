from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from agents.sandbox.capabilities.memory import Memory


def _load_usaspending_agent_module() -> Any:
    try:
        return importlib.import_module(
            "examples.sandbox.extensions.daytona.usaspending_text2sql.agent"
        )
    except SystemExit as exc:
        pytest.skip(str(exc))


def _memory_capability(agent: Any) -> Memory:
    memories = [capability for capability in agent.capabilities if isinstance(capability, Memory)]
    assert len(memories) == 1
    return memories[0]


def test_usaspending_auto_mode_disables_memory_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXAMPLES_INTERACTIVE_MODE", "auto")
    module = _load_usaspending_agent_module()

    memory = _memory_capability(module.build_agent())

    assert memory.read is not None
    assert memory.generate is None


def test_usaspending_prompt_mode_keeps_memory_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXAMPLES_INTERACTIVE_MODE", raising=False)
    module = _load_usaspending_agent_module()

    memory = _memory_capability(module.build_agent())

    assert memory.read is not None
    assert memory.generate is not None


_REPO_ROOT = Path(__file__).resolve().parents[1]

_NON_UTF8_LOCALE_ENV = {
    "LC_ALL": "C",
    "LANG": "C",
    "LC_CTYPE": "C",
    "PYTHONCOERCECLOCALE": "0",
    "PYTHONUTF8": "0",
}

_IMPORT_CHECK = """
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
module = importlib.import_module(
    "examples.sandbox.extensions.daytona.usaspending_text2sql.agent"
)
expected = Path(
    "examples/sandbox/extensions/daytona/usaspending_text2sql/schema/overview.md"
).read_text(encoding="utf-8")
assert module.DEVELOPER_INSTRUCTIONS.startswith(expected), (
    "overview.md content was not decoded as UTF-8"
)
print("IMPORT_OK")
"""

_ROUNDTRIP_CHECK = """
import importlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from agents.extensions.sandbox.daytona.sandbox import DaytonaSandboxSessionState
from agents.sandbox.manifest import Manifest
from agents.sandbox.snapshot import NoopSnapshot

module = importlib.import_module(
    "examples.sandbox.extensions.daytona.usaspending_text2sql.agent"
)
state = DaytonaSandboxSessionState(
    sandbox_id="sb-1",
    name="caf\\u00e9\\u2014session",
    snapshot=NoopSnapshot(type="noop", id="snap-1"),
    manifest=Manifest(),
)
with tempfile.TemporaryDirectory() as tmp:
    module.SESSION_STATE_PATH = Path(tmp) / "state.json"
    module._save_session_state(state)
    loaded = module._load_session_state()
assert loaded is not None, "session state was silently dropped on load"
assert loaded.name == state.name, f"non-ASCII name corrupted: {loaded.name!r}"
print("ROUNDTRIP_OK")
"""


def _run_in_non_utf8_locale(script: str) -> subprocess.CompletedProcess[str]:
    """Run a script in a fresh interpreter forced onto a non-UTF-8 locale.

    This reproduces on every platform the condition of running on a Windows
    machine whose ANSI code page is not UTF-8, where text file calls without
    an explicit encoding use the locale codec instead of UTF-8.
    """
    env = os.environ.copy()
    env.update(_NON_UTF8_LOCALE_ENV)
    return subprocess.run(
        [sys.executable, "-X", "utf8=0", "-c", script],
        cwd=_REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=300,
    )


def test_usaspending_example_imports_under_non_utf8_locale() -> None:
    """The module-level schema read must decode UTF-8 regardless of the locale."""
    result = _run_in_non_utf8_locale(_IMPORT_CHECK)

    assert result.returncode == 0, f"import failed under a non-UTF-8 locale:\n{result.stderr}"
    assert "IMPORT_OK" in result.stdout


def test_session_state_roundtrip_under_non_utf8_locale() -> None:
    """Session state persistence must survive non-ASCII content under any locale."""
    result = _run_in_non_utf8_locale(_ROUNDTRIP_CHECK)

    assert result.returncode == 0, (
        f"session state round-trip failed under a non-UTF-8 locale:\n{result.stderr}"
    )
    assert "ROUNDTRIP_OK" in result.stdout
