"""Context compaction.

When the conversation grows past cfg.compact_limit prompt tokens, the old
part gets replaced by a model-written summary. The system prompt stays, the
most recent cfg.recent_keep messages stay, everything in between becomes one
assistant message that says "here is what happened so far".
"""

import json
import time
from pathlib import Path

from openai import OpenAI

from mini_harness.config import CONFIG
from mini_harness.retry_request import retry_call


class CompactContent:
    def _cut_index(self, messages: list, cfg=CONFIG) -> int:
        """Index where 'old' ends and 'recent' begins.

        Walk back so the cut never lands on a tool result: a tool message
        must stay with the assistant message that requested it, or the API
        rejects the history.
        """
        cut = len(messages) - cfg.recent_keep
        while cut > 1 and messages[cut]["role"] == "tool":
            cut -= 1
        return cut

    def _summary_prompt(self, old: list) -> str:
        """Flatten the old messages into text the summarizer can read."""
        lines = []
        for m in old:
            role = m.get("role")
            if role == "user":
                lines.append(f"[user]: {m.get('content')}")
            elif role == "assistant":
                if m.get("content"):
                    lines.append(f"[assistant]: {m.get('content')}")
                for tc in m.get("tool_calls") or []:
                    fn = tc["function"]
                    lines.append(f"[tool call]: {fn['name']}: {fn['arguments'][:200]}")
            elif role == "tool":
                lines.append(f"[tool result]: {str(m.get('content'))[:200]}")
        history = "\n".join(lines)
        return f"""
Summarize the conversation below into a working summary that a coding agent can
continue from. Cover, in this order:
1. The user's overall goal.
2. What has been done so far.
3. Decisions that were made and why.
4. Files that were read or changed, with their paths.
5. Where the work is right now and what the next step is.

{history}
"""

    def _request_summary(self, client: OpenAI, prompt: str, cfg=CONFIG) -> str:
        response = client.chat.completions.create(
            model=cfg.model_sub,
            messages=[
                {"role": "system", "content": "You summarize a coding agent's conversation history "
                                              "into a concise working summary."},
                {"role": "user", "content": prompt},
            ],
            temperature=cfg.temp_set,
            extra_body=cfg.thinking_sub,
            max_tokens=cfg.max_tokens_sub,
            stream=False,
        )
        return response.choices[0].message.content

    def compact_content(self, client: OpenAI, messages: list, session_path: str | Path | None = None,
                        cfg=CONFIG) -> list:
        cut = self._cut_index(messages, cfg=cfg)
        if cut <= 1:
            return messages            # nothing old enough to summarize
        old = messages[1:cut]          # messages[0] is the system prompt

        print("[compact]: summarizing older history...")
        start = time.time()
        try:
            summary = retry_call(lambda: self._request_summary(client, self._summary_prompt(old), cfg=cfg), cfg=cfg)
        except Exception as e:
            print(f"[compact]: failed, keeping full history this round: {type(e).__name__}: {e}")
            return messages

        # Keep what we threw away, in case someone needs to audit the run.
        if session_path is not None:
            try:
                hist = Path(session_path).with_name("mini_harness_history.jsonl")
                with open(hist, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"ts": time.time(), "removed": old}, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[compact]: could not write history file: {type(e).__name__}: {e}")

        print(f"[compact]: done in {time.time() - start:.1f}s, {len(old)} messages -> 1")
        return [
            messages[0],
            {"role": "assistant", "reasoning_content": "",
             "content": f"Summary of the conversation so far:\n{summary}"},
            *messages[cut:],
        ]


COMPACT = CompactContent()
