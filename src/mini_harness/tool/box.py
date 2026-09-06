"""The tool layer.

Three things live here, in order:

1. Helpers for rendering, writing and fingerprinting files.
2. ToolExecution: takes a tool call from the model, validates it, applies the
   safety gates, runs it, and remembers what the agent has read.
3. The nine tools themselves. Each is a Pydantic input model + a plain
   function, tied together in the TOOLS list at the bottom.
"""

import fnmatch
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Callable

from openai import OpenAI
from pydantic import (
    BaseModel, ConfigDict, Field, StringConstraints, ValidationError,
    field_validator, model_validator,
)

from mini_harness.config import CONFIG
from mini_harness.retry_request import retry_call
from mini_harness.tool.block import CLIP, SUBAGENT, TODO
from mini_harness.tool.path import is_denied, resolve_path, validate_read, validate_write
from mini_harness.tool.tag import HIT, LEVEL, MORE, TAG

# A string that must have something in it after stripping.
Nonblank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

# "    12\t" - the prefix read_file puts in front of every line.
LINE_NO = re.compile(r"^\s*\d+\t")

READ_TOOLS = {"read_file", "grep_file"}
WRITE_TOOLS = {"write_file", "edit_file"}


# ---------------------------------------------------------------------------
# 1. File helpers
# ---------------------------------------------------------------------------

def _safe_mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _split_lines(content: str) -> list[str]:
    """Split on \\n, normalising \\r\\n, and drop the empty tail after a final newline."""
    if not content:
        return []
    lines = content.replace("\r\n", "\n").split("\n")
    if lines and not lines[-1]:
        lines.pop()
    return lines


def _render_lines(path: Path, offset: int | None, limit: int | None, cfg=CONFIG) -> tuple[str, bool]:
    """Render a file cat -n style. Returns (text, reached_end)."""
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"[Binary or non-UTF-8 file, {path.stat().st_size} bytes. Content not displayed]", True

    lines = _split_lines(content)
    total = len(lines)
    if total == 0:
        return "[The file is empty (0 lines)]", True

    start = (offset or 1) - 1
    end = start + limit if limit else total
    window = lines[start:end]
    if not window:
        raise ValueError(
            f"[offset error]: offset {offset} exceeds the file length ({total} lines); "
            f"use an offset between 1 and {total}"
        )

    out, used = [], 0
    for lineno, line in enumerate(window, start + 1):
        rendered = f"{lineno:>6}\t{line}"
        out.append(rendered)
        used += len(rendered)
        if used >= cfg.read_limit:      # stop before we blow the context budget
            break

    last = start + len(out)
    body = "\n".join(out)
    reached_end = last >= total
    if not reached_end:
        body += f"\n\n{MORE}{start + 1}-{last} of {total}, use offset {last + 1} to continue]"
    return body, reached_end


def _atomic_write(path: Path, content: str) -> None:
    """Write to a temp file then rename, so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(content, encoding="utf-8", errors="replace")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _strip_line_no(content: str) -> str:
    """If every non-blank line starts with a 'NN\\t' prefix, remove the prefixes."""
    lines = _split_lines(content)
    if not lines:
        return content
    if all(LINE_NO.match(line) for line in lines if line.strip()):
        return "\n".join(LINE_NO.sub("", line) for line in lines)
    return content


def _key(file_path, cfg=CONFIG) -> str:
    """Canonical absolute path string, used as the dict key for tracked files."""
    return str(resolve_path(file_path, cfg=cfg))


def _digest(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _to_api_tool(tools: list) -> list[dict]:
    """Turn ToolDefinitions into the JSON the chat API expects."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters.model_json_schema(),
            },
        }
        for t in tools
    ]


# Confirmation policies. ToolExecution calls one of these before a risky tool.
def _ask_human(tool_call, cfg=CONFIG) -> bool:
    ans = input(
        f"\nthe agent wants to run {tool_call.function.name}: {tool_call.function.arguments}\n"
        "allow? (yes/no) -> "
    ).strip().lower()
    return ans in cfg.AGREE


def _always_allow(tool_call, cfg=CONFIG) -> bool:
    return True


def _for_sub(tool_call, cfg=CONFIG) -> bool:
    return False   # subagents never get risky tools anyway; this is belt and braces


# ---------------------------------------------------------------------------
# 2. ToolExecution
# ---------------------------------------------------------------------------

@dataclass
class FileRecord:
    """What we know about a file the agent has touched."""

    mtime: float
    digest: str
    level: str      # LEVEL.FULL or LEVEL.PARTIAL
    edits: int = 0


@dataclass(frozen=True)
class ToolItem:
    """Result of one tool call as it goes back to the model."""

    content: str
    ok: bool
    tag: str = ""


class ToolExecution:
    def __init__(self, regis: dict, confirm: Callable, cfg=CONFIG) -> None:
        self.regis = regis          # name -> ToolDefinition
        self.confirm = confirm      # policy for risky tools
        self.last_tool = None       # (name, canonical args) of the previous call
        self.files: dict[str, FileRecord] = {}

    # --- file tracking ---

    def _mark(self, key: str, level: str) -> None:
        """Record that the agent has just seen `key` at `level`. FULL is sticky."""
        p = Path(key)
        try:
            mtime = p.stat().st_mtime
            digest = _digest(p)
        except OSError:
            return
        old = self.files.get(key)
        if level == LEVEL.FULL or (old and old.level == LEVEL.FULL):
            level = LEVEL.FULL
        else:
            level = LEVEL.PARTIAL
        self.files[key] = FileRecord(mtime, digest, level, old.edits if old else 0)

    def _fresh(self, key: str) -> bool:
        """Has the file stayed unchanged since we last marked it? mtime first, hash as tiebreak."""
        rec = self.files.get(key)
        if rec is None:
            return False
        p = Path(key)
        try:
            if p.stat().st_mtime == rec.mtime:
                return True
            return _digest(p) == rec.digest
        except OSError:
            return False

    def _deny(self, key: str, tag: str, msg: str, cfg=CONFIG) -> ToolItem:
        """Refuse a write, but hand back the current file so the model can retry at once."""
        try:
            body, reached_end = _render_lines(Path(key), None, None, cfg=cfg)
            self._mark(key, LEVEL.FULL if reached_end else LEVEL.PARTIAL)
            return ToolItem(f"[{tag}]: {msg}\n\nCurrent content of {key}:\n{body}", False, tag)
        except Exception as e:
            return ToolItem(
                f"[{tag}]: {msg}\n\n[Could not display the file: {type(e).__name__}. "
                "Read it with read_file before retrying]",
                False, tag,
            )

    def _gate(self, name: str, args, cfg=CONFIG) -> ToolItem | None:
        """The read-before-write rules. Returns a refusal, or None to proceed."""
        if not cfg.track_files or name not in WRITE_TOOLS:
            return None
        file_path = getattr(args, "file_path", None)
        if file_path is None:
            return None

        key = _key(file_path, cfg=cfg)
        exists = Path(key).is_file()

        if name == "edit_file" and cfg.edit_require_read:
            if not exists:
                return None     # edit_file itself will raise FileNotFoundError
            if key not in self.files:
                return self._deny(key, TAG.NEED_READ,
                                  "You have not read this file in this session. Read it first, then retry.")
            if not self._fresh(key):
                return self._deny(key, TAG.STALE,
                                  "This file changed after you last read it. Review the current content below, then retry.")

        if name == "write_file" and cfg.write_require_read:
            if not exists:
                return None
            if not args.overwrite:
                return self._deny(key, TAG.EXISTS,
                                  "This file already exists. write_file creates new files. To replace it entirely, "
                                  "read it in full first and pass overwrite=true; to change part of it, use edit_file.")
            rec = self.files.get(key)
            if rec is None or rec.level != LEVEL.FULL:
                return self._deny(key, TAG.NEED_FULL,
                                  "Overwriting destroys the whole file, so you must have read it end to end first. "
                                  "The current content is below; retry after reviewing it.")
            if not self._fresh(key):
                return self._deny(key, TAG.STALE,
                                  "This file changed after you last read it. Review the current content below, then retry.")
        return None

    def _record(self, name: str, args, content: str, cfg=CONFIG) -> str:
        """After a successful call, update the file table. May append a note to the result."""
        if not cfg.track_files:
            return content

        if name == "read_file":
            self._mark(_key(args.file_path, cfg=cfg), LEVEL.PARTIAL if MORE in content else LEVEL.FULL)

        elif name == "grep_file":
            for hit in set(HIT.findall(content)):
                self._mark(hit, LEVEL.PARTIAL)

        elif name in WRITE_TOOLS:
            key = _key(args.file_path, cfg=cfg)
            old = self.files.get(key)
            edits = old.edits if old else 0
            level = LEVEL.FULL if name == "write_file" else (old.level if old else LEVEL.FULL)
            self._mark(key, level)
            self.files[key].edits = edits + 1
            n = self.files[key].edits
            if cfg.thrash_notice and n >= cfg.thrash_notice:
                content += (f"\n\n[You have modified this file {n} times without the task passing. "
                            "Consider re-reading it in full, or reconsidering the approach.]")
        return content

    # --- the main entry point ---

    def execute_tool(self, tool_call, cfg=CONFIG) -> ToolItem:
        name = tool_call.function.name
        tool = self.regis.get(name)
        if tool is None:
            known = "\n".join(self.regis)
            return ToolItem(f"[{TAG.UNKNOWN_TOOL}]: unknown tool {name}. Available tools:\n{known}",
                            False, TAG.UNKNOWN_TOOL)

        # 1. validate arguments against the Pydantic model
        try:
            args = tool.parameters.model_validate_json(tool_call.function.arguments)
        except ValidationError as e:
            return ToolItem(f"[{TAG.INVALID_ARGS}]: {e}", False, TAG.INVALID_ARGS)

        # 2. refuse an exact repeat of the previous call (a common stuck-loop symptom)
        current = (name, args.model_dump_json())
        if self.last_tool == current:
            return ToolItem(f"[{TAG.DEDUP}]: you just called {name} with the same arguments. "
                            "Use a different tool or different arguments.", False, TAG.DEDUP)

        # 3. read-before-write gates
        gate = self._gate(name, args, cfg=cfg)
        if gate is not None:
            return gate

        # 4. human confirmation for risky tools
        if tool.risky and not self.confirm(tool_call, cfg=cfg):
            return ToolItem(f"[{TAG.DENIED}]: the user denied this call. Do not retry it; "
                            "explain what you needed and propose an alternative.", False, TAG.DENIED)
        self.last_tool = current

        # 5. run it
        try:
            result = tool.function(args)
        except Exception as e:
            return ToolItem(f"[{TAG.EXECUTE_FAILED}]: {type(e).__name__}: {e}",
                            False, f"{TAG.EXECUTE_FAILED}:{type(e).__name__}")

        raw = result if isinstance(result, str) else json.dumps(result)
        raw = self._record(name, args, raw, cfg=cfg)
        return ToolItem(raw, True, TAG.SUCCESS)


def log_tool(tool_call, res: ToolItem, prefix: str = "", cfg=CONFIG) -> None:
    """One line per tool call on the console."""
    arguments = tool_call.function.arguments
    if len(arguments) > 100:
        arguments = arguments[:100] + "\n....clipped at 100 chars"
    suffix = "" if res.ok else f" -> failed: {res.tag}"
    print(f"{prefix}{tool_call.function.name}: {arguments}{suffix}")


# ---------------------------------------------------------------------------
# 3. The tools
# ---------------------------------------------------------------------------

# --- glob_file ---

class GlobFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pattern: Nonblank = Field(description=(
        'File name pattern. Without a "/" it searches every subdirectory '
        '(e.g. "*.py"); with one it is relative to path (e.g. "src/*.py").'))
    path: str = Field(".", description="Directory to search in.")


def glob_file(inp: GlobFileInput, cfg=CONFIG) -> str:
    root = validate_read(inp.path, cfg=cfg)
    pattern = inp.pattern if "/" in inp.pattern else "**/" + inp.pattern

    matches = []
    for rel in glob.glob(pattern, root_dir=root, recursive=True):
        full = root / rel
        real = full.resolve()
        if not real.is_file():
            continue
        if cfg.guard_read and not real.is_relative_to(cfg.work_space):
            continue    # a symlink pointing outside the workspace
        if is_denied(real, cfg=cfg):
            continue
        matches.append(full)

    matches = sorted(set(matches), key=_safe_mtime, reverse=True)   # newest first
    return "\n".join(str(m) for m in matches) if matches else "no matches"


# --- grep_file ---

class GrepFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pattern: Nonblank = Field(description="Regular expression to search for.")
    path: str = Field(".", description="File or directory to search.")
    glob: str | None = Field(None, description='Only search files matching this name pattern, e.g. "*.py".')

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as e:
            raise ValueError(f"invalid regex: {e}")
        return value


def grep_file(inp: GrepFileInput, cfg=CONFIG) -> str:
    root = validate_read(inp.path, cfg=cfg)
    regex = re.compile(inp.pattern)

    if root.is_file():
        targets = [root]
    else:
        targets = []
        for dirpath, dirnames, filenames in os.walk(root):
            # prune hidden and denied directories in place so os.walk skips them
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in cfg.deny_dir]
            for name in filenames:
                real = (Path(dirpath) / name).resolve()
                if cfg.guard_read and not real.is_relative_to(cfg.work_space):
                    continue
                if is_denied(real, cfg=cfg):
                    continue
                if inp.glob and not fnmatch.fnmatch(name, inp.glob):
                    continue
                targets.append(real)

    hits, truncated = [], False
    for path in targets:
        if truncated:
            break
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, 1):
                    if regex.search(line):
                        hits.append(f"[{path}]: {lineno}: {line.rstrip()}")
                        if len(hits) >= cfg.max_hits:
                            truncated = True
                            break
        except (OSError, UnicodeDecodeError):
            continue
    if truncated:
        hits.append(f"\n..... truncated at {cfg.max_hits} hits")
    return "\n".join(hits) if hits else "no matches"


# --- read_file ---

class ReadFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_path: Nonblank = Field(description="Path of the file to read.")
    offset: int | None = Field(None, ge=1, description=(
        "1-based line number to start from. Same numbering as the line numbers "
        "in read_file and grep_file output."))
    limit: int | None = Field(None, ge=1, description="Number of lines to read.")


def read_file(inp: ReadFileInput, cfg=CONFIG) -> str:
    path = validate_read(inp.file_path, cfg=cfg)
    if not path.is_file():
        raise FileNotFoundError(f"[not found]: {path}")
    if inp.offset is None and inp.limit is None:
        size = path.stat().st_size
        if size >= cfg.max_read_size:
            raise ValueError(f"[oversize]: {path} is {size} bytes, limit is {cfg.max_read_size}. "
                             "Read it with offset/limit.")
    body, _ = _render_lines(path, inp.offset, inp.limit, cfg=cfg)
    return body


# --- write_file ---

class WriteFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_path: Nonblank = Field(description="Path of the file to create.")
    content: str = Field(description="Full content to write.")
    overwrite: bool = Field(False, description=(
        "Set true only to replace an existing file entirely. Requires having read it in full first. "
        "Leave false to create a new file."))


def write_file(inp: WriteFileInput, cfg=CONFIG) -> str:
    path = validate_write(inp.file_path, cfg=cfg)
    existed = path.is_file()
    _atomic_write(path, inp.content)
    n = len(_split_lines(inp.content))
    verb = "Overwrote" if existed else "Created"
    return f"{verb} {path} ({n} lines, {len(inp.content)} chars)"


# --- edit_file ---

class EditFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_path: Nonblank = Field(description="Path of the file to edit.")
    old_string: str = Field(description=(
        "Exact text to replace. Must appear in the file verbatim, including indentation. "
        "Do not include the line-number prefixes shown by read_file."))
    new_string: str = Field(description="Replacement text. Do not include line-number prefixes.")
    replace_all: bool = Field(False, description="Replace every occurrence instead of requiring exactly one.")

    @model_validator(mode="after")
    def _differs(self):
        if self.old_string == self.new_string:
            raise ValueError("old_string and new_string are identical")
        return self


def edit_file(inp: EditFileInput, cfg=CONFIG) -> str:
    path = validate_write(inp.file_path, cfg=cfg)
    if not path.is_file():
        raise FileNotFoundError(f"[not found]: {path}")
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"[encoding]: {path} is not valid UTF-8 (byte {e.start}: {e.reason}); "
                         "edit_file would corrupt it.")

    old = inp.old_string
    if old not in content:
        # Common model mistake: it pasted read_file output, line numbers included.
        stripped = _strip_line_no(old)
        if stripped != old and stripped in content:
            if any(LINE_NO.match(line) for line in _split_lines(inp.new_string)):
                raise ValueError("[line numbers]: old_string contained line-number prefixes and was fixed, "
                                 "but new_string contains them too. Resend both without the 'NN\\t' prefix.")
            if stripped == inp.new_string:
                raise ValueError("old_string and new_string are identical once line numbers are removed")
            old = stripped
        else:
            raise ValueError(f"[not found]: old_string does not appear in {path}")

    count = content.count(old)
    if not inp.replace_all and count > 1:
        raise ValueError(f"[ambiguous]: old_string appears {count} times in {path}; "
                         "add context to make it unique or pass replace_all=true")

    start_line = content[:content.index(old)].count("\n") + 1
    old_n, new_n = len(_split_lines(old)), len(_split_lines(inp.new_string))
    replaced = count if inp.replace_all else 1
    new_content = content.replace(old, inp.new_string) if inp.replace_all else content.replace(old, inp.new_string, 1)
    _atomic_write(path, new_content)

    total = len(_split_lines(new_content))
    head = f"Replaced {replaced} occurrence(s) in {path}\n({old_n} -> {new_n} lines, file now {total} lines)."
    if cfg.diff_echo_lines == 0:
        return head

    # Echo the new lines with their real numbers so the model can keep editing without re-reading.
    shown = _split_lines(inp.new_string)[:cfg.diff_echo_lines]
    body = "\n".join(f"{start_line + i:>6}\t{line}" for i, line in enumerate(shown))
    if new_n > cfg.diff_echo_lines:
        body += f"\n[showing first {cfg.diff_echo_lines} of {new_n} changed lines]"
    return head + "\n" + body


# --- run_bash ---

class RunBashInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: Nonblank = Field(description="Shell command to run on the host, from the workspace root.")


def run_bash(inp: RunBashInput, cfg=CONFIG) -> str:
    try:
        result = subprocess.run(
            inp.command, shell=True, capture_output=True, text=True,
            timeout=cfg.bash_timeout, env=cfg.bash_env, cwd=cfg.work_space,
        )
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"[timeout]: command exceeded {cfg.bash_timeout}s: {e}")

    parts = []
    if result.stdout:
        parts.append(result.stdout)
    if result.stderr:
        parts.append(f"[stderr]: {result.stderr}")
    if result.returncode != 0:
        parts.append(f"[exit code: {result.returncode}]")
    content = "\n".join(parts) if parts else "no output"
    if len(content) >= cfg.bash_limit:
        total = len(content)
        content = content[:cfg.bash_limit] + (
            f"\n[.....truncated at {cfg.bash_limit} of {total} chars. Narrow the command "
            "(grep/head/tail) or redirect to a file and read it with offset/limit]")
    return content


# --- run_sandbox ---

class RunSandboxInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: Nonblank = Field(description=(
        "Shell command inside an isolated Python container. sandbox/ is mounted at /workspace; "
        "use paths relative to it."))
    timeout_seconds: int = Field(30, ge=1, le=300, strict=True, description="Maximum runtime in seconds.")


def run_sandbox(inp: RunSandboxInput, cfg=CONFIG) -> str:
    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("run_sandbox requires Docker. Install/start Docker, then: docker pull python:3.12-slim")

    root = cfg.work_space.resolve()
    workspace = cfg.sandbox_dir
    if workspace.is_symlink() or not workspace.resolve().is_relative_to(root):
        raise PermissionError("sandbox/ must be a real directory inside the workspace")
    workspace.mkdir(parents=True, exist_ok=True)
    workspace = workspace.resolve()
    if "," in str(workspace):
        raise ValueError("Docker --mount cannot handle a path containing a comma")

    name = "mini-harness-" + uuid.uuid4().hex
    # A tiny Python runner inside the container enforces the timeout and maps the exit code.
    runner = (
        "import subprocess,sys\n"
        "try:\n"
        " p=subprocess.run(sys.argv[1],shell=True,timeout=int(sys.argv[2]))\n"
        " sys.exit(p.returncode if p.returncode >= 0 else 128-p.returncode)\n"
        "except subprocess.TimeoutExpired:\n"
        ' print("[sandbox command timed out]",flush=True)\n'
        " sys.exit(124)\n"
    )
    command = [
        docker, "run", "--rm", "--pull=never", "--name", name,
        "--label", "mini-harness.sandbox=true",
        "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
        "--pids-limit=64", "--memory=256m", "--memory-swap=256m", "--cpus=1", "--log-driver=none",
        "--ulimit", "fsize=16777216:16777216",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m,mode=1777",
        "--mount", f"type=bind,src={workspace},dst=/workspace", "--workdir", "/workspace",
        "--env", "HOME=/tmp", "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--entrypoint", "python", "python:3.12-slim", "-u", "-c", runner,
        inp.command, str(inp.timeout_seconds),
    ]

    output = bytearray()
    clipped = False

    def drain(pipe):
        nonlocal clipped
        with pipe:
            while chunk := pipe.read(8192):
                room = max(0, cfg.bash_limit - len(output))
                output.extend(chunk[:room])
                clipped |= len(chunk) > room

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, env=cfg.bash_env, start_new_session=True)
    reader = threading.Thread(target=drain, args=(process.stdout,), daemon=True)
    reader.start()
    try:
        process.wait(timeout=inp.timeout_seconds + 10)
    except subprocess.TimeoutExpired as e:
        raise TimeoutError("Docker sandbox startup or execution timed out") from e
    finally:
        try:
            cleanup = subprocess.run([docker, "rm", "-f", name], capture_output=True, timeout=5, env=cfg.bash_env)
            if cleanup.returncode and b"No such container" not in cleanup.stderr:
                raise RuntimeError(f"Could not remove sandbox container {name}: "
                                   + cleanup.stderr.decode(errors="replace"))
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            reader.join(timeout=1)

    text = output.decode("utf-8", errors="replace") or "no output"
    if clipped:
        text += f"\n[output truncated at {cfg.bash_limit} bytes]"
    if process.returncode == 124:
        raise TimeoutError(f"Sandbox exceeded {inp.timeout_seconds}s.\n{text}")
    if process.returncode:
        raise RuntimeError(f"Sandbox exited with code {process.returncode}.\n{text}")
    return text


# --- run_todo ---

class StatusItem(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"


class TodoItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: Nonblank = Field(description="The task, imperative form (e.g. 'fix the bug').")
    activeForm: Nonblank = Field(description="Present-continuous form (e.g. 'fixing the bug').")
    status: StatusItem = Field(description="pending, in_progress or completed.")


class RunTodoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[TodoItem] = Field(description="The full todo list.", min_length=1, max_length=20)

    @model_validator(mode="after")
    def _one_in_progress(self):
        if sum(1 for t in self.items if t.status == StatusItem.in_progress) > 1:
            raise ValueError("only one item may be in_progress at a time")
        return self


def run_todo(inp: RunTodoInput, cfg=CONFIG) -> str:
    return TODO.update(inp.items)


# --- run_subagent ---

class AgentType(str, Enum):
    explore_agent = "explore_agent"
    coding_agent = "coding_agent"
    planning_agent = "planning_agent"


class RunSubAgentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_description: Nonblank = Field(description="One line shown to the user while the subagent runs.")
    prompt: Nonblank = Field(description="The full instructions for the subagent.")
    agent_type: AgentType = Field(description=(
        "explore_agent (find and read files), coding_agent (write and edit files), "
        "planning_agent (write a plan)."))


def run_subagent(inp: RunSubAgentInput, cfg=CONFIG) -> str:
    """A second, smaller agent loop: fewer tools, no streaming, no confirmation, returns a summary."""
    spec = SUBAGENT.agent_table.get(inp.agent_type.value)
    if spec is None:
        raise ValueError(f"[invalid subagent]: {inp.agent_type.value}")

    allowed = [t for t in TOOLS if t.name in spec["tools"]]
    api_tools = _to_api_tool(allowed)
    regis = {t.name: t for t in allowed}

    messages = [
        {"role": "system", "content": (
            f"You are a {inp.agent_type.value}. Your responsibility is to {spec['description']}.\n"
            f"{spec['prompt']}\n"
            "Finish with a summary of what you found or did.")},
        {"role": "user", "content": inp.prompt},
    ]

    start = time.time()
    print(f"[{inp.agent_type.value}]: {inp.task_description}")
    client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=cfg.base_url, max_retries=0)
    executor = ToolExecution(regis, _for_sub, cfg=cfg)
    calls = 0

    for _ in range(cfg.max_turns_sub):
        response = retry_call(lambda: client.chat.completions.create(
            model=cfg.model_sub, messages=messages, tools=api_tools,
            max_tokens=cfg.max_tokens_sub, extra_body=cfg.thinking_sub,
            temperature=cfg.temp_set, stream=False,
        ), cfg=cfg)
        message = response.choices[0].message
        if not message.tool_calls:
            messages.append(message.model_dump(exclude_none=True))
            print(f"[{inp.agent_type.value}]: {inp.task_description} -- {calls} tools -- {time.time() - start:.1f}s")
            return message.content
        messages.append(message.model_dump(exclude_none=True))
        for tool_call in message.tool_calls:
            res = executor.execute_tool(tool_call, cfg=cfg)
            calls += 1
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": CLIP.clip(res.content, cfg=cfg)})
    return "[subagent]: ran out of turns before finishing; the task is incomplete"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: type[BaseModel]
    function: Callable
    risky: bool     # True -> ToolExecution asks the confirm policy first


TOOLS = [
    ToolDefinition("glob_file", "Find files by name pattern.", GlobFileInput, glob_file, False),
    ToolDefinition("grep_file", "Search file contents with a regular expression.", GrepFileInput, grep_file, False),
    ToolDefinition("read_file",
                   "Read a file. Output uses cat -n style line numbers, which are for reference only and must "
                   "never be copied into edit_file. Without offset/limit the whole file is returned, truncated "
                   "with a continuation hint if it is very large.",
                   ReadFileInput, read_file, False),
    ToolDefinition("write_file",
                   "Create a new file. To replace an existing file entirely you must have read it in full first "
                   "and pass overwrite=true; for partial changes use edit_file instead.",
                   WriteFileInput, write_file, False),
    ToolDefinition("edit_file",
                   "Replace a specific string in an existing file. You must have read the file first, and it must "
                   "not have changed since. old_string must match the file exactly and must not contain "
                   "line-number prefixes. On success the tool echoes the resulting lines with their real line numbers.",
                   EditFileInput, edit_file, False),
    ToolDefinition("run_bash", "Run a shell command on the host.", RunBashInput, run_bash, True),
    ToolDefinition("run_sandbox",
                   "Run code in a disposable Docker Python 3.12 container with no network and resource limits. "
                   "Only sandbox/ is shared, as /workspace; writes there persist. Requires Docker and a locally "
                   "pulled python:3.12-slim image. Prefer this for running generated code; run_bash executes on the host.",
                   RunSandboxInput, run_sandbox, True),
    ToolDefinition("run_todo", "Create or update the todo list.", RunTodoInput, run_todo, False),
    ToolDefinition("run_subagent", "Delegate a separable subtask to a subagent.", RunSubAgentInput, run_subagent, True),
]

_names = {t.name for t in TOOLS}
if not (READ_TOOLS | WRITE_TOOLS) <= _names:
    raise RuntimeError(f"[tool name drift]: {(READ_TOOLS | WRITE_TOOLS) - _names} not in TOOLS")
