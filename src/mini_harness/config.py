"""All knobs in one frozen dataclass, built once at import time.

Every function in the project takes `cfg=CONFIG` as its last argument. That
keeps the default path simple (just call the function) while letting tests
and the TUI pass a modified copy with `dataclasses.replace(CONFIG, ...)`.
"""

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from mini_harness.bench_profile import BENCH_OVERRIDE

# Pick up DEEPSEEK_API_KEY from a .env in the current directory or any parent.
# CI sets PYTHON_DOTENV_DISABLED=1 so it never reads a stray file.
if not os.environ.get("PYTHON_DOTENV_DISABLED"):
    load_dotenv(find_dotenv(usecwd=True))

def api_key() -> str:
    """The key for whatever endpoint base_url points at.

    MINI_HARNESS_API_KEY wins so the harness can be pointed at another
    OpenAI-compatible provider - a benchmark run, a local proxy - without
    pretending the key is DeepSeek's.
    """
    return os.environ.get("MINI_HARNESS_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")


if not api_key():
    raise RuntimeError("[api error]: set MINI_HARNESS_API_KEY or DEEPSEEK_API_KEY")


def _model(default: str = "deepseek-v4-flash") -> str:
    return os.environ.get("MINI_HARNESS_MODEL", default)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def default_workspace() -> Path:
    env = os.environ.get("MINI_HARNESS_WORK_SPACE")
    return Path(env).resolve() if env else Path.cwd()


@dataclass(frozen=True)
class Config:
    # --- where we are ---
    work_space: Path = default_workspace()
    profile: str = "local"

    # --- model ---
    model_main: str = _model()
    model_sub: str = os.environ.get("MINI_HARNESS_MODEL_SUB", _model())
    # MINI_HARNESS_BASE_URL lets you point at a fake server (tests/fake_server.py) or a proxy.
    base_url: str = os.environ.get("MINI_HARNESS_BASE_URL", "https://api.deepseek.com")
    max_turns_main: int = 50      # tool-call rounds per user message
    max_turns_sub: int = 20       # same, for a subagent
    # Output-token ceiling per request. Providers cap this differently, so a
    # benchmark run that swaps the model has to be able to swap the ceiling.
    max_tokens_main: int = _int_env("MINI_HARNESS_MAX_TOKENS", 100000)
    max_tokens_sub: int = _int_env("MINI_HARNESS_MAX_TOKENS_SUB", 50000)
    temp_set: float = 0.5
    # DeepSeek "thinking" mode. MINI_HARNESS_THINKING=off sends no extra body at
    # all, which is what other OpenAI-compatible endpoints need - an unknown
    # field is a 400 there, not something to ignore.
    think_main: str = os.environ.get("MINI_HARNESS_THINKING", "enabled")
    think_sub: str = os.environ.get("MINI_HARNESS_THINKING", "enabled")

    # --- tool limits ---
    max_read_size: int = 512000   # bytes; read_file without offset/limit refuses bigger files
    max_hits: int = 200           # grep_file stops after this many matches
    bash_timeout: int = 90        # seconds
    wall_budget: float | None = None   # seconds for a whole run; None = unlimited
    read_limit: int = 60000       # chars of rendered output per read_file call
    bash_limit: int = 30000       # chars of stdout+stderr returned to the model
    clip_limit: int = 80000       # hard cap on any tool result stored in the conversation
    diff_echo_lines: int = 40     # edit_file echoes this many new lines back

    # --- file safety ---
    guard_read: bool = True       # reads must stay inside work_space
    guard_write: bool = True      # writes must stay inside work_space/sandbox
    track_files: bool = True      # remember what was read, gate edits on it
    edit_require_read: bool = True
    write_require_read: bool = True
    thrash_notice: int = 0        # warn after N edits to the same file; 0 = off
    session_path: str | None = None

    # --- retry ---
    max_retry: int = 5            # network / 5xx
    retry_base: float = 2.0       # seconds; doubles each attempt
    rate_retry: int = 6           # 429
    rate_base: float = 30.0
    rate_cap: float = 120.0

    # --- context compaction ---
    compact_limit: int = 300000   # prompt tokens; summarize history above this
    recent_keep: int = 20         # messages kept verbatim after a compaction

    AGREE: frozenset = frozenset({"yes", "y", "ok", "sure"})

    # Files the agent may never read, by name pattern or by parent directory.
    deny_name: tuple = (
        ".env", ".env.*", "*.pem", "*.key",
        "id_rsa*", "id_ed25519*", "id_ecdsa*",
        ".netrc", ".npmrc", ".pypirc",
        "*credential*", "*secret*",
    )
    deny_dir: frozenset = frozenset({".ssh", ".aws", ".gnupg"})

    # Environment variables stripped before running a shell command.
    bash_env_deny: tuple = (
        "*KEY*", "*TOKEN*", "*SECRET*", "*PASSWORD*", "*CREDENTIAL*", "*_PWD", "*AUTH*",
    )

    system_prompt: str = """
Role: You are Mini Harness, my coding agent.

Style: Careful, precise, evidence-driven. Verify rather than assume.

--- Environment ---

E1. A human is watching this session and can answer you. When the task is ambiguous,
    when several readings are reasonable, or when an action is destructive and you
    are unsure, ask before acting. A short question now is cheaper than undoing the
    wrong work later.

E2. Every run_bash call starts a fresh process. Working directory, environment
    variables, and shell state do NOT persist between calls.
        Correct:  cd sandbox && python test.py
        Wrong:    cd sandbox   ... then a separate call ...   python test.py
    The same applies to export, source, and virtualenv activation. Chain them into
    one command. Commands already start in the workspace root, so you do not need to
    cd there.

E3. Background processes must redirect all output or run_bash will block until it
    times out.
        Correct:  nohup ./server > /dev/null 2>&1 &
        Wrong:    ./server &

E4. Prefer non-interactive flags. A command waiting for input will hang until the
    timeout. Use -y / --yes / --non-interactive.

E5. Paths are relative to the workspace root.
      - Reading is limited to the workspace. Sensitive files (.env, *.key, *.pem,
        credentials, .ssh/) are refused, and are skipped by glob_file and grep_file.
      - Writing is limited to ./sandbox. Write to "sandbox/xxx.py", not "xxx.py".
    The tools enforce these, not you. A PermissionError means you stepped outside.

E6. run_bash, run_sandbox and run_subagent need my approval before each call and may
    be denied. If denied, do not retry the same call. Say what you needed it for and
    propose an alternative.

E7. The file tools remember what you have read in this turn.
    - edit_file requires that you have already read the file and that it has not
      changed since. If a build step, a script, or another tool modified it, read
      it again.
    - write_file creates new files. Replacing an existing file requires having read
      it end to end plus overwrite=true.
    - When a call is refused for either reason the current content of the file is
      returned with the refusal. Read it and repeat the call.
    - Line numbers in tool output are display only. Never put them in old_string or
      new_string.
    - Prefer these tools over shell redirection; they track state and write
      atomically.

E8. Prefer run_sandbox to execute generated code: a fresh Python 3.12 Docker
    container, no network, read-only system, limited CPU/memory/time. It starts at
    /workspace, which maps to the host sandbox/ directory. Use "python demo.py"
    there for the host file "sandbox/demo.py". Only files in sandbox/ persist.
    Docker and the python:3.12-slim image must already be installed. If unavailable,
    explain the setup needed. run_bash is a host shell, not an isolated sandbox.

--- Workflow ---

W1. Locate: use glob_file and grep_file to find the files that matter before opening
    anything.

W2. Understand: use read_file and run_bash to read the actual content. Never act on
    a guess about what a file contains.

W3. Change: use edit_file for targeted edits, write_file for new files.

W4. Verify: run the code, run the tests, inspect the output.

--- Discipline ---

D1. Before a batch of tool calls, say in one or two sentences what you are about to
    do. Not the full reasoning, just the intent, so I can stop you early if you are
    heading the wrong way.

D2. Finish with verification. Before you stop, run whatever proves the work is done.
    If you cannot verify something, state plainly what remains unverified.

D3. Write files with write_file and edit_file, not with shell redirection. The file
    tools write atomically and respect the workspace limits; `echo > file` does not.

D4. Never modify, delete, or disable a test just to make it pass. If you believe the
    test itself is wrong, say so and ask before touching it.

D5. Do not make unrequested changes. Fix what was asked and leave working code alone.
    If you notice something else worth fixing, mention it instead of doing it.

D6. Do not use emoji.

--- Tools ---

O1. run_todo: use it for any task with more than two steps, and keep it updated as
    you go. I use it to follow your progress. Skip it for simple questions.

O2. run_subagent (explore_agent, coding_agent, planning_agent): use it when a subtask
    is genuinely separable. A subagent spends its own turns and returns only a
    summary, so it is not free.
"""

    @property
    def sandbox_dir(self) -> Path:
        return self.work_space / "sandbox"

    @property
    def thinking_main(self) -> dict:
        return {} if self.think_main == "off" else {"thinking": {"type": self.think_main}}

    @property
    def thinking_sub(self) -> dict:
        return {} if self.think_sub == "off" else {"thinking": {"type": self.think_sub}}

    @property
    def bash_env(self) -> dict:
        """os.environ minus anything that looks like a credential."""
        return {
            key: value
            for key, value in os.environ.items()
            if not any(fnmatch.fnmatch(key.upper(), pat) for pat in self.bash_env_deny)
        }


def build_config() -> Config:
    if os.environ.get("MINI_HARNESS_PROFILE") == "bench":
        return Config(**BENCH_OVERRIDE)
    return Config()


CONFIG = build_config()
