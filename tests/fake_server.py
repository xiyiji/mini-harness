"""A scripted stand-in for the DeepSeek chat-completions endpoint.

It speaks just enough of the OpenAI streaming protocol for the agent loop to
run end to end without a network or an API key. Each request pops the next
scripted reply; a reply is either text or a list of tool calls.

    with FakeModel([tool("read_file", file_path="sandbox/a.py"), "done"]) as base_url:
        ...

Also usable by hand:  python tests/fake_server.py  then
    DEEPSEEK_API_KEY=x MINI_HARNESS_BASE_URL=http://127.0.0.1:8765 uv run mini-harness --task "hi"
"""

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def tool(name: str, **arguments) -> dict:
    return {"name": name, "arguments": json.dumps(arguments)}


class FakeModel:
    def __init__(self, script: list, port: int = 0) -> None:
        self.script = list(script)
        self.requests: list[dict] = []      # every request body we received, for assertions
        self.port = port
        self.server = None
        self.thread = None

    # --- streaming chunks ---

    def _chunk(self, delta: dict, finish: str | None = None, usage: dict | None = None) -> bytes:
        body = {
            "id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": int(time.time()),
            "model": "fake", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if usage is not None:
            body["usage"] = usage
        return f"data: {json.dumps(body)}\n\n".encode()

    def _stream(self, reply) -> list[bytes]:
        usage = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
        out = [self._chunk({"role": "assistant", "content": ""})]
        if isinstance(reply, str):
            for i in range(0, len(reply), 8):
                out.append(self._chunk({"content": reply[i:i + 8]}))
            out.append(self._chunk({}, finish="stop"))
        else:
            calls = reply if isinstance(reply, list) else [reply]
            for i, c in enumerate(calls):
                out.append(self._chunk({"tool_calls": [{
                    "index": i, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                    "function": {"name": c["name"], "arguments": ""}}]}))
                out.append(self._chunk({"tool_calls": [{"index": i, "function": {"arguments": c["arguments"]}}]}))
            out.append(self._chunk({}, finish="tool_calls"))
        out.append(self._chunk({}, usage=usage))
        out.append(b"data: [DONE]\n\n")
        return out

    def _json(self, reply) -> bytes:
        message = {"role": "assistant", "content": None}
        finish = "stop"
        if isinstance(reply, str):
            message["content"] = reply
        else:
            calls = reply if isinstance(reply, list) else [reply]
            message["tool_calls"] = [{"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                                      "function": c} for c in calls]
            finish = "tool_calls"
        return json.dumps({
            "id": "chatcmpl-fake", "object": "chat.completion", "created": int(time.time()), "model": "fake",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        }).encode()

    # --- server ---

    def __enter__(self) -> str:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                fake.requests.append(body)
                reply = fake.script.pop(0) if fake.script else "(script exhausted)"
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for chunk in fake._stream(reply):
                        self.wfile.write(chunk)
                        self.wfile.flush()
                else:
                    payload = fake._json(reply)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}"

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()


if __name__ == "__main__":
    demo = FakeModel([
        tool("run_todo", items=[{"content": "say hello", "activeForm": "saying hello", "status": "in_progress"}]),
        tool("write_file", file_path="sandbox/hello.py", content="print('hello from the fake model')\n"),
        tool("run_bash", command="python sandbox/hello.py"),
        "I wrote sandbox/hello.py and ran it. Done.",
    ], port=8765)
    with demo as url:
        print(f"fake model listening on {url}  (Ctrl+C to stop)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
