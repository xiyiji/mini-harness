"""End-to-end tests of the agent loop against the fake model server."""

import json
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("DEEPSEEK_API_KEY", "dummy")

from mini_harness.agent import DeepSeekAgent          # noqa: E402
from mini_harness.compact import COMPACT              # noqa: E402
from mini_harness.config import CONFIG                # noqa: E402
from mini_harness.tool.box import TOOLS               # noqa: E402
from mini_harness.tool.tag import OUTCOME             # noqa: E402
from tests.fake_server import FakeModel, tool         # noqa: E402


def make_cfg(tmp_path: Path, base_url: str, **over):
    (tmp_path / "sandbox").mkdir(exist_ok=True)
    cfg = replace(CONFIG, work_space=tmp_path, base_url=base_url, **over)
    for t in TOOLS:
        t.function.__defaults__ = (cfg,)
    return cfg


def teardown_module(module):
    for t in TOOLS:
        t.function.__defaults__ = (CONFIG,)


def test_full_loop_write_run_answer(tmp_path):
    script = [
        tool("write_file", file_path="sandbox/hello.py", content="print(6 * 7)\n"),
        tool("run_bash", command="python3 sandbox/hello.py"),
        "The answer is 42.",
    ]
    with FakeModel(script) as url:
        cfg = make_cfg(tmp_path, url)
        agent = DeepSeekAgent(TOOLS, cfg=cfg)
        result = agent.run_task("write and run hello", cfg=cfg)

    assert result.outcome == OUTCOME.COMPLETED
    assert result.turns == 3 and result.calls == 2 and result.ok == 2
    assert result.calls_by_tool == {"write_file": 1, "run_bash": 1}
    assert (tmp_path / "sandbox" / "hello.py").exists()

    # the conversation on disk is well formed: every tool_call has a tool result
    saved = json.loads((tmp_path / "session.json").read_text())
    roles = [m["role"] for m in saved]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert "42" in saved[5]["content"]      # the bash output went back to the model


def test_tool_failure_is_reported_not_raised(tmp_path):
    script = [
        tool("edit_file", file_path="sandbox/none.py", old_string="a", new_string="b"),
        "could not edit",
    ]
    with FakeModel(script) as url:
        cfg = make_cfg(tmp_path, url)
        result = DeepSeekAgent(TOOLS, cfg=cfg).run_task("edit", cfg=cfg)
    assert result.outcome == OUTCOME.COMPLETED
    assert result.ok == 0 and list(result.failed_by_tag)[0].startswith("execute_failed")


def test_exhausted_when_model_never_stops(tmp_path):
    script = [tool("glob_file", pattern=f"*{i}.py") for i in range(5)]
    with FakeModel(script) as url:
        cfg = make_cfg(tmp_path, url, max_turns_main=3)
        result = DeepSeekAgent(TOOLS, cfg=cfg).run_task("loop", cfg=cfg)
    assert result.outcome == OUTCOME.EXHAUSTED and result.turns == 3


def test_compaction_replaces_old_history(tmp_path):
    with FakeModel(["a summary of everything so far"]) as url:
        cfg = make_cfg(tmp_path, url, recent_keep=2)
        messages = [{"role": "system", "content": "sys"}]
        for i in range(6):
            messages.append({"role": "user", "content": f"u{i}"})
            messages.append({"role": "assistant", "content": f"a{i}"})
        from openai import OpenAI
        client = OpenAI(api_key="x", base_url=url, max_retries=0)
        new = COMPACT.compact_content(client, messages, tmp_path / "session.json", cfg=cfg)

    assert new[0]["role"] == "system"
    assert "a summary of everything" in new[1]["content"]
    assert [m["content"] for m in new[2:]] == ["u5", "a5"]
    # the removed part was archived
    hist = (tmp_path / "mini_harness_history.jsonl").read_text()
    assert "u0" in hist and "a4" in hist


def test_compaction_never_cuts_between_tool_call_and_result():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": "1", "function": {"name": "x", "arguments": "{}"}}]})
    messages += [{"role": "tool", "tool_call_id": "1", "content": "r"}] * 3
    messages.append({"role": "assistant", "content": "done"})
    cfg = replace(CONFIG, recent_keep=3)          # naive cut would land on a tool message
    cut = COMPACT._cut_index(messages, cfg=cfg)
    assert messages[cut]["role"] == "assistant"
