"""Offline tests for the tool layer. No API key or network needed
(DEEPSEEK_API_KEY just has to be set to anything).

Run:  DEEPSEEK_API_KEY=dummy uv run pytest -q
"""

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("DEEPSEEK_API_KEY", "dummy")

from mini_harness.config import CONFIG                        # noqa: E402
from mini_harness.tool.box import TOOLS, ToolExecution, _always_allow  # noqa: E402
from mini_harness.tool.tag import TAG                         # noqa: E402


def call(name: str, **kwargs):
    """Fake the object the OpenAI SDK hands us for a tool call."""
    return SimpleNamespace(id="c1", function=SimpleNamespace(name=name, arguments=json.dumps(kwargs)))


@pytest.fixture
def ws(tmp_path: Path):
    """A throwaway workspace with one file in it, plus a matching config and executor."""
    (tmp_path / "sandbox").mkdir()
    (tmp_path / "sandbox" / "hello.py").write_text("print('hi')\nprint('bye')\n")
    (tmp_path / ".env").write_text("SECRET=1\n")
    cfg = replace(CONFIG, work_space=tmp_path)
    regis = {t.name: t for t in TOOLS}
    ex = ToolExecution(regis, _always_allow, cfg=cfg)
    # the tool functions read cfg from their own default; patch it in for the test
    for t in TOOLS:
        t.function.__defaults__ = (cfg,)
    yield tmp_path, cfg, ex
    for t in TOOLS:
        t.function.__defaults__ = (CONFIG,)


def test_unknown_tool_and_bad_args(ws):
    _, cfg, ex = ws
    assert ex.execute_tool(call("nope"), cfg=cfg).tag == TAG.UNKNOWN_TOOL
    assert ex.execute_tool(call("read_file", nope=1), cfg=cfg).tag == TAG.INVALID_ARGS


def test_read_then_dedup(ws):
    _, cfg, ex = ws
    r1 = ex.execute_tool(call("read_file", file_path="sandbox/hello.py"), cfg=cfg)
    assert r1.ok and "     1\tprint('hi')" in r1.content
    r2 = ex.execute_tool(call("read_file", file_path="sandbox/hello.py"), cfg=cfg)
    assert r2.tag == TAG.DEDUP


def test_edit_requires_read(ws):
    root, cfg, ex = ws
    r = ex.execute_tool(call("edit_file", file_path="sandbox/hello.py", old_string="hi", new_string="yo"), cfg=cfg)
    assert r.tag == TAG.NEED_READ
    assert "print('hi')" in r.content          # refusal hands back the file
    # the refusal itself counts as a read, so the retry goes through
    r = ex.execute_tool(call("edit_file", file_path="sandbox/hello.py", old_string="hi", new_string="yo"), cfg=cfg)
    assert r.ok, r.content
    assert (root / "sandbox" / "hello.py").read_text().startswith("print('yo')")


def test_edit_detects_stale(ws):
    root, cfg, ex = ws
    ex.execute_tool(call("read_file", file_path="sandbox/hello.py"), cfg=cfg)
    p = root / "sandbox" / "hello.py"
    p.write_text("changed\n")
    os.utime(p, (1, 1))   # force a different mtime
    r = ex.execute_tool(call("edit_file", file_path="sandbox/hello.py", old_string="changed", new_string="x"), cfg=cfg)
    assert r.tag == TAG.STALE


def test_edit_strips_pasted_line_numbers(ws):
    _, cfg, ex = ws
    ex.execute_tool(call("read_file", file_path="sandbox/hello.py"), cfg=cfg)
    r = ex.execute_tool(call("edit_file", file_path="sandbox/hello.py",
                             old_string="     1\tprint('hi')", new_string="print('fixed')"), cfg=cfg)
    assert r.ok, r.content


def test_write_gates(ws):
    _, cfg, ex = ws
    r = ex.execute_tool(call("write_file", file_path="sandbox/hello.py", content="x"), cfg=cfg)
    assert r.tag == TAG.EXISTS
    r = ex.execute_tool(call("write_file", file_path="sandbox/new.py", content="x = 1\n"), cfg=cfg)
    assert r.ok and "Created" in r.content
    r = ex.execute_tool(call("write_file", file_path="outside.py", content="x"), cfg=cfg)
    assert r.tag.startswith(TAG.EXECUTE_FAILED) and "PermissionError" in r.tag


def test_read_guards(ws):
    _, cfg, ex = ws
    r = ex.execute_tool(call("read_file", file_path=".env"), cfg=cfg)
    assert "PermissionError" in r.tag
    r = ex.execute_tool(call("read_file", file_path="../../etc/passwd"), cfg=cfg)
    assert "PermissionError" in r.tag


def test_glob_and_grep_skip_denied_files(ws):
    _, cfg, ex = ws
    r = ex.execute_tool(call("glob_file", pattern="*"), cfg=cfg)
    assert "hello.py" in r.content and ".env" not in r.content
    r = ex.execute_tool(call("grep_file", pattern="print|SECRET"), cfg=cfg)
    assert "hello.py]: 1:" in r.content and "SECRET" not in r.content
    # grep marks the file as partially read, which is enough for edit_file
    r = ex.execute_tool(call("edit_file", file_path="sandbox/hello.py", old_string="bye", new_string="ciao"), cfg=cfg)
    assert r.ok, r.content


def test_run_bash_strips_secrets_from_env(ws):
    _, cfg, ex = ws
    os.environ["MY_TOKEN"] = "leak"
    r = ex.execute_tool(call("run_bash", command="echo [$MY_TOKEN] [$DEEPSEEK_API_KEY] && pwd"), cfg=cfg)
    assert r.ok and "leak" not in r.content and "dummy" not in r.content
    assert str(cfg.work_space) in r.content


def test_run_todo(ws):
    _, cfg, ex = ws
    items = [{"content": "a", "activeForm": "doing a", "status": "completed"},
             {"content": "b", "activeForm": "doing b", "status": "in_progress"}]
    r = ex.execute_tool(call("run_todo", items=items), cfg=cfg)
    assert r.ok and "[x] a" in r.content and "[>] doing b" in r.content and "1 / 2" in r.content
    items[0]["status"] = "in_progress"
    r = ex.execute_tool(call("run_todo", items=items), cfg=cfg)
    assert r.tag == TAG.INVALID_ARGS


def test_tool_schemas_are_valid_json_schema():
    from mini_harness.tool.box import _to_api_tool
    api = _to_api_tool(TOOLS)
    assert len(api) == 9
    for t in api:
        assert t["function"]["parameters"]["type"] == "object"
        assert t["function"]["parameters"]["additionalProperties"] is False
