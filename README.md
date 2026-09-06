# mini-harness

A small coding-agent harness in Python. About 1,800 lines: an agent loop with
streaming, nine tools defined with Pydantic, read-before-write gating on file
edits, context compaction, request retries, a Docker sandbox, subagents, and
an optional terminal UI.

Stack: Python 3.12, `openai` SDK (DeepSeek endpoint), Pydantic 2, Textual for
the TUI, uv for packaging. Docker is optional and only used by `run_sandbox`.

This is a from-scratch reimplementation of the design in
[mini-harness/mini-harness](https://github.com/mini-harness/mini-harness).
The module layout and behaviour follow that project; the code was written
fresh, with tests and a fake model server added so the whole loop can be
exercised offline.

## Layout

```
src/mini_harness/
  agent.py           the loop: ask the model, run its tool calls, repeat until it answers
  compact.py         summarise old history when the prompt grows past a token limit
  config.py          one frozen Config, built at import; every function takes cfg=CONFIG
  bench_profile.py   overrides and system prompt for unattended benchmark runs
  retry_request.py   exponential backoff for network, 5xx and 429
  main.py            CLI: interactive, or --task for one unattended run
  tool/
    box.py           ToolExecution (validate, dedupe, gate, confirm, run) and the nine tools
    block.py         todo list, output clipper, subagent catalogue
    path.py          read/write path guards
    tag.py           shared constants
tui.py               Textual UI; runs the agent in a worker subprocess over JSON lines
tests/
  fake_server.py     a scripted stand-in for the chat-completions endpoint
  test_tools.py      tool layer, offline
  test_agent.py      agent loop end to end against the fake server
```

## Tools

| Tool | What it does | Asks first |
| --- | --- | --- |
| `glob_file` | find files by name pattern, newest first | |
| `grep_file` | regex search across files | |
| `read_file` | read with `cat -n` line numbers, offset/limit for large files | |
| `write_file` | create a file; overwriting requires a full prior read | |
| `edit_file` | replace an exact string; requires a prior read and an unchanged file | |
| `run_bash` | shell command on the host, credentials stripped from the environment | yes |
| `run_sandbox` | command in a throwaway Docker container, no network, `sandbox/` mounted | yes |
| `run_todo` | keep a checklist the user can follow | |
| `run_subagent` | delegate to an explore / coding / planning subagent with fewer tools | yes |

A tool is a Pydantic input model, a function, and a `ToolDefinition`:

```python
from pydantic import BaseModel, ConfigDict, Field
from mini_harness.agent import DeepSeekAgent
from mini_harness.tool.box import TOOLS, ToolDefinition

class CountWordsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, description="Text to count words in.")

def count_words(args: CountWordsInput) -> str:
    return str(len(args.text.split()))

count_words_tool = ToolDefinition(
    name="count_words", description="Count whitespace-separated words.",
    parameters=CountWordsInput, function=count_words, risky=False,
)

agent = DeepSeekAgent([*TOOLS, count_words_tool])
```

`model_json_schema()` describes the tool to the model; `model_validate_json()`
checks the arguments before the function runs. Invalid calls go back to the
model as an error it can correct.

## File safety

Reads are limited to the workspace and skip credential-looking files
(`.env`, `*.pem`, `.ssh/`, ...). Writes are limited to `./sandbox`. The
executor remembers what the agent has read: `edit_file` on a file that was
never read, or that changed since, is refused and the current content is
returned with the refusal so the next attempt can succeed. Overwriting with
`write_file` requires having read the file end to end. The bench profile
turns all of this off.

## Get started

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh     # once
git clone <this repo> && cd mini-harness
uv python install 3.12
uv sync --locked
export DEEPSEEK_API_KEY="your-api-key"

uv run tui.py                 # terminal UI
uv run --locked mini-harness  # plain REPL
uv run --locked mini-harness --task "add a test for parse()" --telemetry-out run.json
```

TUI keys: `Enter` send, `Ctrl+N` new session, `F2` sessions, `Ctrl+E` expand
tool cards, `Ctrl+C` cancel, `Ctrl+Q` quit. Sessions live in `.local/`.

For `run_sandbox`, install Docker and pull the image once:

```sh
docker pull python:3.12-slim
```

## Running without an API key

`tests/fake_server.py` speaks enough of the OpenAI streaming protocol to drive
the agent through a scripted conversation:

```sh
DEEPSEEK_API_KEY=x uv run python tests/fake_server.py          # terminal 1
DEEPSEEK_API_KEY=x MINI_HARNESS_BASE_URL=http://127.0.0.1:8765 \
  uv run mini-harness --task "write a hello script and run it"  # terminal 2
```

The same server backs the test suite:

```sh
DEEPSEEK_API_KEY=x uv run pytest -q
```

## Benchmark profile

`MINI_HARNESS_PROFILE=bench` raises the turn and timeout limits, disables the
workspace guards, and swaps in a system prompt for unattended runs. `--task`
prints one JSON line prefixed with `####MINI_HARNESS_RUN####` and exits with
0 (completed), 3 (turns exhausted), 4 (wall budget), 1 (error), or 130
(interrupted), so a driver script can grep and branch on it.

## Configuration

Everything is a field on `Config` in `config.py`. Environment overrides:
`MINI_HARNESS_WORK_SPACE`, `MINI_HARNESS_BASE_URL`, `MINI_HARNESS_PROFILE`,
`MINI_HARNESS_WALL_BUDGET`. A `.env` in the working directory is loaded for
`DEEPSEEK_API_KEY`.
