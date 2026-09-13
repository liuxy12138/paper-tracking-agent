from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent_main


def test_main_validates_ask_before_building_agent(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["agent_main.py", "ask"],
    )

    build_called = False

    def fake_build_agent(*args, **kwargs):
        nonlocal build_called
        build_called = True
        raise AssertionError("_build_agent should not be called for empty ask input")

    monkeypatch.setattr(agent_main, "_build_agent", fake_build_agent)

    with pytest.raises(SystemExit, match="Please provide --question for the ask command."):
        agent_main.main()

    assert build_called is False
