"""Shared string constants.

Everything that gets compared by name across modules lives here, so a typo
shows up as an AttributeError instead of a silent mismatch.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Tag:
    """Why a tool call did not succeed. Empty string means it did."""

    SUCCESS: str = ""
    INVALID_ARGS: str = "invalid_args"     # arguments failed Pydantic validation
    DENIED: str = "denied"                 # human said no to a risky tool
    EXECUTE_FAILED: str = "execute_failed" # the tool function raised
    DEDUP: str = "duplicate"               # same tool + same args as the previous call
    UNKNOWN_TOOL: str = "unknown_tool"     # model asked for a tool we never registered
    NEED_READ: str = "need_read"           # edit_file before read_file
    STALE: str = "stale"                   # file changed since it was last read
    EXISTS: str = "exists"                 # write_file on an existing file without overwrite
    NEED_FULL: str = "need_full"           # overwrite requested but file was only partially read


@dataclass(frozen=True)
class Level:
    """How much of a file the agent has seen."""

    PARTIAL: str = "partial"
    FULL: str = "full"


@dataclass(frozen=True)
class Outcome:
    """How a whole agent run ended."""

    COMPLETED: str = "completed"       # model replied without a tool call
    INTERRUPTED: str = "interrupted"   # Ctrl+C
    ERROR: str = "error"               # unexpected exception
    EXHAUSTED: str = "exhausted"       # hit max_turns
    TIMEOUT: str = "timeout"           # hit wall_budget


TAG = Tag()
LEVEL = Level()
OUTCOME = Outcome()

# Line prefix that main.py prints before the JSON telemetry of a --task run.
# A benchmark driver can grep for it in the log.
MARK = "####MINI_HARNESS_RUN####"

# Marker read_file appends when it did not reach the end of the file.
# ToolExecution uses it to decide whether the read was FULL or PARTIAL.
MORE = "[Showing lines "

# Matches the "[/abs/path]: 12: " prefix of a grep_file hit; group 1 is the path.
HIT = re.compile(r"^\[(.+?)\]: \d+: ", re.M)
