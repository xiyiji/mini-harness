"""Small stateful helpers the tools lean on: the todo list, the output
clipper, and the subagent catalogue.

They live apart from box.py so that box.py stays "just tools" and these
can be swapped or extended without touching the executor.
"""

from mini_harness.config import CONFIG


class TodoManager:
    """Holds the agent's current todo list and renders it as a checklist."""

    def __init__(self) -> None:
        self.items: list = []

    def update(self, items: list) -> str:
        self.items = items
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "no items"
        lines = []
        for item in self.items:
            if item.status == "completed":
                lines.append(f"[x] {item.content}")
            elif item.status == "in_progress":
                lines.append(f"[>] {item.activeForm}")
            else:
                lines.append(f"[ ] {item.content}")
        done = sum(1 for t in self.items if t.status == "completed")
        lines.append(f"{done} / {len(self.items)} completed")
        return "\n".join(lines)


class Clip:
    """Last line of defence for the context window: no tool result may exceed clip_limit."""

    def clip(self, content: str, cfg=CONFIG) -> str:
        if len(content) < cfg.clip_limit:
            return content
        return content[:cfg.clip_limit] + f"\n.....clipped at {cfg.clip_limit} of {len(content)} chars"


class SubAgent:
    """Which tools and which system prompt each subagent type gets."""

    def __init__(self) -> None:
        self.agent_table = {
            "explore_agent": {
                "description": "explore, read and find files and file content relevant to the task",
                "tools": ["read_file", "glob_file", "grep_file"],
                "prompt": """
You are an explore agent. Your job is to read and analyze files.
Step 1: locate. Use glob_file and grep_file to find the files that matter.
Step 2: read. Use read_file to read them, in full where it matters.
Step 3: report. Summarize what you found: which files, what they do, what is relevant to the task.
""",
            },
            "coding_agent": {
                "description": "create, write and modify files to complete a coding task",
                "tools": ["read_file", "glob_file", "grep_file", "write_file", "edit_file"],
                "prompt": """
You are a coding agent. Your job is to write and edit files.
Step 1: understand. Use read_file, glob_file and grep_file to see the current state before changing anything.
Step 2: consider. Think through more than one way to do it and pick the simplest that works.
Step 3: write. Use write_file for new files and edit_file for targeted changes.
Step 4: report. Say what you changed and what remains unverified.
""",
            },
            "planning_agent": {
                "description": "read, analyze and write a plan for completing the task",
                "tools": ["read_file", "glob_file", "grep_file", "write_file", "edit_file"],
                "prompt": """
You are a planning agent. Your job is to produce a plan, not to execute it.
Step 1: analyze. Use read_file, glob_file and grep_file to understand the goal and the code.
Step 2: write. Save the plan as a markdown file with write_file, then return the analysis and the file path.
""",
            },
        }


TODO = TodoManager()
CLIP = Clip()
SUBAGENT = SubAgent()
