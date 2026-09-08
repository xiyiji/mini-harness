"""The agent loop.

    user message
        -> ask the model (streaming)
        -> did it call tools?
              yes: run each one, append results, go back to "ask the model"
              no:  done, that reply is the answer

Everything else in this file is bookkeeping around that loop: saving the
session, compacting when it grows, retrying, counting tokens, and handling
the model getting cut off or the human pressing Ctrl+C.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import LengthFinishReasonError, OpenAI

from mini_harness.compact import COMPACT
from mini_harness.config import CONFIG, api_key
from mini_harness.retry_request import retry_call
from mini_harness.tool.block import CLIP
from mini_harness.tool.box import (
    ToolExecution, _always_allow, _ask_human, _atomic_write, _to_api_tool, log_tool,
)
from mini_harness.tool.tag import MARK, OUTCOME


@dataclass(frozen=True)
class Result:
    """Telemetry for one run (one user message, or one --task)."""

    outcome: str
    calls: int                 # tool calls made
    turns: int                 # model requests made
    ok: int                    # tool calls that succeeded
    failed_by_tag: dict        # {tag: count}
    calls_by_tool: dict        # {tool name: count}
    last_prompt: int           # prompt tokens of the final request
    prompt_total: int
    completion_total: int
    wall: float                # seconds
    err: str = ""


class DeepSeekAgent:
    def __init__(self, tools: list, cfg=CONFIG) -> None:
        self.tools = _to_api_tool(tools)              # what the API sees
        self.regis = {t.name: t for t in tools}       # what we execute
        self.system = [{"role": "system", "content": cfg.system_prompt}]
        self.message = list(self.system)
        self.session_memory = cfg.session_path or cfg.work_space / "session.json"
        self.last_prompt_tokens = 0
        self.last_usage = None
        self.last_reasoning = ""
        self.printed = ""       # what we have echoed to the console during this request

    # ------------------------------------------------------------------
    # session persistence
    # ------------------------------------------------------------------

    def _load_memory(self, cfg=CONFIG) -> list:
        path = Path(self.session_memory)
        if not path.exists():
            return self.message
        ans = input("\nFound a saved session. Continue it? (yes/no) -> ").strip().lower()
        if ans not in cfg.AGREE:
            return self.message
        try:
            self.message = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"[load error]: {e}; starting a new session")
        return self.message

    def _save_memory(self, quiet: bool = False, cfg=CONFIG) -> None:
        try:
            _atomic_write(Path(self.session_memory), json.dumps(self.message, indent=2, ensure_ascii=False))
        except (TypeError, ValueError, OSError) as e:
            print(f"[save failed]: {e}")
            return
        if not quiet:
            print("[saved]")

    # ------------------------------------------------------------------
    # one request to the model
    # ------------------------------------------------------------------

    def _request_agent(self, client: OpenAI, cfg=CONFIG):
        """Stream one completion. Returns (response, truncated_text).

        response is None when the model hit max_tokens; truncated_text then
        holds whatever content it produced before the cut.

        The `printed` / `buffer` dance handles a reconnect: if retry_call
        re-runs this after a dropped stream, the new stream starts from zero.
        We only print the part we have not shown yet, and if the new text
        diverges from what was already on screen we say so.
        """
        buffer = ""
        reasoning = ""
        streaming = False
        truncated = None

        with client.chat.completions.stream(
            model=cfg.model_main,
            tools=self.tools,
            messages=self.message,
            temperature=cfg.temp_set,
            extra_body=cfg.thinking_main,
            max_tokens=cfg.max_tokens_main,
            stream_options={"include_usage": True},
        ) as stream:
            for event in stream:
                if event.type == "content.delta":
                    if streaming:
                        print(event.delta, end="", flush=True)
                        self.printed += event.delta
                        continue
                    buffer += event.delta
                    n = min(len(buffer), len(self.printed))
                    if buffer[:n] != self.printed[:n]:
                        print("\n---- [connection lost, the text above is superseded] ----")
                        print(buffer, end="", flush=True)
                        self.printed = buffer
                        streaming = True
                    elif len(buffer) > len(self.printed):
                        print(buffer[len(self.printed):], end="", flush=True)
                        self.printed = buffer
                        streaming = True
                elif event.type == "chunk":
                    usage = getattr(event.chunk, "usage", None)
                    if usage is not None:
                        self.last_usage = usage
                    if event.chunk.choices:
                        piece = getattr(event.chunk.choices[0].delta, "reasoning_content", None)
                        if piece:
                            reasoning += piece
                            print(piece, end="", flush=True)

            try:
                response = stream.get_final_completion()
                response.choices[0].message.reasoning_content = reasoning
            except LengthFinishReasonError as e:
                truncated = buffer
                completion = getattr(e, "completion", None)
                usage = getattr(completion, "usage", None) if completion is not None else None
                if usage is not None:
                    self.last_usage = usage
                response = None

        if not streaming and self.printed:
            # the whole reply was already on screen from a previous attempt
            print("\n---- [reconnected, the text above is superseded] ----")
            if buffer:
                print(buffer, end="", flush=True)
            self.printed = buffer
        self.last_reasoning = reasoning
        return response, truncated

    def _fill_interrupted(self, tool_calls) -> None:
        """Ctrl+C mid-batch: every tool_call needs a tool message or the API rejects the history."""
        done = {m["tool_call_id"] for m in self.message if m["role"] == "tool"}
        for tc in tool_calls:
            if tc.id not in done:
                self.message.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": f"[interrupted]: {tc.function.name} did not run"})

    # ------------------------------------------------------------------
    # the loop
    # ------------------------------------------------------------------

    def _run_turn(self, client: OpenAI, executor: ToolExecution, cfg=CONFIG) -> Result:
        start = time.time()
        outcome = OUTCOME.ERROR
        turns = calls = ok = last_prompt = prompt_total = completion_total = 0
        by_tag: dict = {}
        by_tool: dict = {}
        err = ""

        try:
            for _ in range(cfg.max_turns_main):
                if cfg.wall_budget is not None and time.time() - start >= cfg.wall_budget:
                    outcome = OUTCOME.TIMEOUT
                    print("[timeout]: wall budget exhausted")
                    break
                turns += 1
                self.printed = ""

                # compaction happens before the request, based on the previous request's size
                if self.last_prompt_tokens >= cfg.compact_limit:
                    self.message = COMPACT.compact_content(client, self.message, self.session_memory, cfg=cfg)
                    self._save_memory(quiet=True)
                    self.last_prompt_tokens = 0

                response, truncated = retry_call(lambda: self._request_agent(client, cfg=cfg), cfg=cfg)

                if truncated is not None:
                    # model hit max_tokens: keep what it said, nudge it, and go again
                    note = ("\n\n[Your previous response was cut off at the output token limit. "
                            "Be more concise, or take an action instead of continuing to reason.]")
                    self.message.append({"role": "assistant", "content": truncated + note,
                                         "reasoning_content": self.last_reasoning})
                    usage = self.last_usage
                    if usage is not None:
                        self.last_prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                        last_prompt = self.last_prompt_tokens
                        prompt_total += self.last_prompt_tokens
                        completion_total += getattr(usage, "completion_tokens", 0) or 0
                    print(f"[ctx]: {last_prompt} / {cfg.compact_limit} tokens [TRUNCATED]")
                    self._save_memory(quiet=True)
                    continue

                if self.printed:
                    print()     # end the streamed line before status output
                if response.usage:
                    self.last_prompt_tokens = response.usage.prompt_tokens
                    last_prompt = response.usage.prompt_tokens
                    prompt_total += response.usage.prompt_tokens
                    completion_total += response.usage.completion_tokens
                    print(f"[ctx]: {last_prompt} / {cfg.compact_limit} tokens, out {response.usage.completion_tokens}")

                message = response.choices[0].message
                if not message.tool_calls:
                    # plain reply = the model is done
                    outcome = OUTCOME.COMPLETED
                    self.message.append(message.model_dump(exclude_none=True))
                    self._save_memory(quiet=True)
                    break

                print()
                self.message.append(message.model_dump(exclude_none=True))
                try:
                    for tool_call in message.tool_calls:
                        res = executor.execute_tool(tool_call, cfg=cfg)
                        calls += 1
                        name = tool_call.function.name
                        by_tool[name] = by_tool.get(name, 0) + 1
                        if res.ok:
                            ok += 1
                            if name == "run_todo":
                                print(f"\n-=-=-=-=-= Todo -=-=-=-=-=\n{res.content}")
                        else:
                            by_tag[res.tag] = by_tag.get(res.tag, 0) + 1
                        self.message.append({"role": "tool", "tool_call_id": tool_call.id,
                                             "content": CLIP.clip(res.content, cfg=cfg)})
                        log_tool(tool_call, res, cfg=cfg)
                except KeyboardInterrupt:
                    self._fill_interrupted(message.tool_calls)
                    raise
                self._save_memory(quiet=True)
            else:
                outcome = OUTCOME.EXHAUSTED
                self._save_memory(quiet=True)
                print("[exhausted]: hit max_turns without a final answer")
        except KeyboardInterrupt:
            outcome = OUTCOME.INTERRUPTED
            print("\n[interrupted]")
            self._save_memory(quiet=True)
        except Exception as e:
            outcome = OUTCOME.ERROR
            err = f"{type(e).__name__}: {e}"
            print(f"[run failed]: {err}")
            self._save_memory(quiet=True)

        return Result(
            outcome=outcome, calls=calls, turns=turns, ok=ok,
            failed_by_tag=by_tag, calls_by_tool=by_tool,
            last_prompt=last_prompt, prompt_total=prompt_total, completion_total=completion_total,
            wall=time.time() - start, err=err,
        )

    # ------------------------------------------------------------------
    # entry points
    # ------------------------------------------------------------------

    def _client(self, cfg=CONFIG) -> OpenAI:
        # max_retries=0: we do our own retrying in retry_call, with printing and backoff we control
        return OpenAI(api_key=api_key(), base_url=cfg.base_url, max_retries=0)

    def run_task(self, task: str, cfg=CONFIG) -> Result:
        """Unattended: one task, no confirmation prompts, returns telemetry."""
        self.message = list(self.system)
        self.message.append({"role": "user", "content": task})
        result = self._run_turn(self._client(cfg), ToolExecution(self.regis, _always_allow, cfg=cfg), cfg=cfg)
        self._save_memory(quiet=True)
        return result

    def dump_run(self, result: Result, task: str, path: str | None = None, cfg=CONFIG) -> None:
        payload = {
            "profile": cfg.profile, "model": cfg.model_main, "thinking": cfg.think_main,
            "task": task, "max_turns_main": cfg.max_turns_main, **asdict(result),
        }
        print(f"{MARK}{json.dumps(payload, ensure_ascii=False)}")
        if path:
            try:
                _atomic_write(Path(path), json.dumps(payload, indent=2, ensure_ascii=False))
            except OSError as e:
                print(f"[dump failed]: {e}")

    def run(self, cfg=CONFIG) -> None:
        """Interactive: read a line, run a turn, repeat. Ctrl+C twice or 'exit' to quit."""
        self.message = self._load_memory(cfg=cfg)
        client = self._client(cfg)
        pending_exit = False
        while True:
            try:
                user_input = input("\n> ")
            except KeyboardInterrupt:
                if not pending_exit:
                    print("\n(press Ctrl+C again to exit)")
                    pending_exit = True
                    continue
                print("\nBye!")
                self._save_memory(quiet=True)
                break
            except EOFError:
                print("\nBye!")
                self._save_memory(quiet=True)
                break

            if user_input.strip().lower() in {"quit", "exit", "bye"}:
                self._save_memory()
                print("Bye!")
                break
            if not user_input.strip():
                continue

            pending_exit = False
            self.message.append({"role": "user", "content": user_input})
            result = self._run_turn(client, ToolExecution(self.regis, _ask_human, cfg=cfg), cfg=cfg)
            self._save_memory(quiet=True)
            print(
                f"\n[{result.outcome}] turns={result.turns} calls={result.calls} ok={result.ok} "
                f"tools={result.calls_by_tool} failed={result.failed_by_tag} "
                f"prompt={result.prompt_total} completion={result.completion_total} "
                f"wall={result.wall:.1f}s{(' err=' + result.err) if result.err else ''}"
            )
