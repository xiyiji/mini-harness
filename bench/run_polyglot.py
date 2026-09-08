"""Run mini-harness against the Exercism exercises used by the polyglot benchmark.

Each exercise is a stub file, a test file the agent may not touch, and an
instructions page. The harness gets the instructions and the file names; the
score is whether the exercise's own test suite passes afterwards. Nothing is
graded by a model, so the number means the same thing every time it is run.

    git clone --depth 1 https://github.com/Aider-AI/polyglot-benchmark
    uv run python bench/run_polyglot.py --repo ../polyglot-benchmark \
        --languages python go rust --per-language 15 --budget-usd 4

The `.meta` directory of every exercise holds a reference solution and is
removed from the copy the agent sees.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

MARK = "####MINI_HARNESS_RUN####"
ROOT = Path(__file__).resolve().parents[1]

# dollars per million tokens: (input, output)
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "deepseek-v4-flash": (0.28, 0.42),
}


@dataclass
class Language:
    """How to find the solution file and how to run the exercise's tests."""

    name: str
    solution: str  # glob for the file the agent must write, inside the exercise
    tests: list[str]  # globs for files it must not touch
    command: list[str]
    timeout: int = 300


LANGUAGES = {
    "python": Language("python", "*.py", ["*_test.py"], ["python", "-m", "pytest", "-q", "-x"]),
    "go": Language("go", "*.go", ["*_test.go"], ["go", "test", "./..."]),
    # Exercism marks every Rust test after the first `#[ignore]`; a plain
    # `cargo test` runs one test and reports success. Include them, as the
    # published benchmark does, or the score measures nothing.
    "rust": Language(
        "rust",
        "src/lib.rs",
        ["tests/*.rs"],
        ["cargo", "test", "--offline", "-q", "--", "--include-ignored"],
    ),
}

PROMPT = """Implement the exercise below.

{instructions}

Write your solution in `{solution}`. The file already exists as a stub with the
expected function and class names; keep those names.

The test suite is `{tests}`. Read it - it is the specification, including the
edge cases. Do not modify it, and do not modify any other test file.

Run `{command}` from the workspace root until it passes. When every test passes,
stop. Do not ask questions; nobody is watching.
"""


@dataclass
class Outcome:
    language: str
    exercise: str
    passed: bool
    agent_outcome: str = ""
    turns: int = 0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    agent_seconds: float = 0.0
    test_seconds: float = 0.0
    touched_tests: bool = False
    note: str = ""
    tools: dict = field(default_factory=dict)


def price(model: str, prompt: int, completion: int) -> float:
    key = next((k for k in PRICES if model.startswith(k)), None)
    if key is None:
        return 0.0
    cin, cout = PRICES[key]
    return prompt / 1e6 * cin + completion / 1e6 * cout


def prepare(src: Path, work: Path, lang: Language) -> tuple[Path, dict[str, bytes]]:
    """A clean copy of the exercise without the reference solution."""
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(src, work)
    shutil.rmtree(work / ".meta", ignore_errors=True)
    # Fingerprint the test files so a run that edits them can be marked, not trusted.
    guarded: dict[str, bytes] = {}
    for pattern in lang.tests:
        for path in work.glob(pattern):
            guarded[str(path.relative_to(work))] = path.read_bytes()
    return work, guarded


def instructions_for(exercise: Path) -> str:
    parts = []
    for name in ("instructions.md", "instructions.append.md"):
        page = exercise / ".docs" / name
        if page.exists():
            parts.append(page.read_text(errors="replace"))
    return "\n\n".join(parts).strip()


def run_agent(work: Path, task: str, env: dict[str, str], timeout: int) -> tuple[dict, float]:
    started = time.monotonic()
    proc = subprocess.run(
        ["uv", "run", "--locked", "mini-harness", "--task", task],
        cwd=ROOT,
        env={
            **env,
            "MINI_HARNESS_WORK_SPACE": str(work),
            "MINI_HARNESS_SESSION": str(work / ".session.json"),
        },
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    elapsed = time.monotonic() - started
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(MARK):
            return json.loads(line[len(MARK) :]), elapsed
    return {"outcome": "no-telemetry", "err": (proc.stderr or proc.stdout)[-400:]}, elapsed


def run_tests(work: Path, lang: Language) -> tuple[bool, float, str]:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            lang.command,
            cwd=work,
            capture_output=True,
            text=True,
            timeout=lang.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, time.monotonic() - started, "test timeout"
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    return proc.returncode == 0, time.monotonic() - started, tail[-1][:160] if tail else ""


def pick(repo: Path, language: str, count: int, seed: int) -> list[Path]:
    practice = repo / language / "exercises" / "practice"
    exercises = sorted(p for p in practice.iterdir() if p.is_dir())
    random.Random(seed).shuffle(exercises)
    return exercises[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="a clone of Aider-AI/polyglot-benchmark")
    parser.add_argument("--languages", nargs="+", default=["python", "go", "rust"])
    parser.add_argument("--per-language", type=int, default=15)
    parser.add_argument("--model", default=os.environ.get("MINI_HARNESS_MODEL", "claude-haiku-4-5"))
    parser.add_argument("--budget-usd", type=float, default=5.0)
    parser.add_argument("--agent-timeout", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--work", default="/tmp/polyglot-runs")
    parser.add_argument("--out", default=str(Path(__file__).with_name("report.md")))
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    workroot = Path(args.work)
    workroot.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MINI_HARNESS_PROFILE": "bench", "MINI_HARNESS_MODEL": args.model}

    results: list[Outcome] = []
    spent = 0.0
    for language in args.languages:
        lang = LANGUAGES[language]
        for exercise in pick(repo, language, args.per_language, args.seed):
            if spent >= args.budget_usd:
                print(f"budget reached (${spent:.2f}); stopping")
                break
            work, guarded = prepare(exercise, workroot / language / exercise.name, lang)
            solution = next(iter(sorted(work.glob(lang.solution))), None)
            if solution is None:
                continue
            tests = ", ".join(f"`{name}`" for name in guarded) or "the test files"
            task = PROMPT.format(
                instructions=instructions_for(work),
                solution=solution.relative_to(work),
                tests=tests,
                command=" ".join(lang.command),
            )
            try:
                telemetry, agent_seconds = run_agent(work, task, env, args.agent_timeout)
            except subprocess.TimeoutExpired:
                telemetry, agent_seconds = {"outcome": "driver-timeout"}, float(args.agent_timeout)

            touched = any(
                not (work / name).exists() or (work / name).read_bytes() != body
                for name, body in guarded.items()
            )
            passed, test_seconds, note = run_tests(work, lang)
            cost = price(
                args.model,
                telemetry.get("prompt_total", 0),
                telemetry.get("completion_total", 0),
            )
            spent += cost
            results.append(
                Outcome(
                    language=language,
                    exercise=exercise.name,
                    # A run that edited the tests has not solved anything.
                    passed=passed and not touched,
                    agent_outcome=str(telemetry.get("outcome", "")),
                    turns=int(telemetry.get("turns", 0)),
                    calls=int(telemetry.get("calls", 0)),
                    prompt_tokens=int(telemetry.get("prompt_total", 0)),
                    completion_tokens=int(telemetry.get("completion_total", 0)),
                    cost_usd=round(cost, 5),
                    agent_seconds=round(agent_seconds, 1),
                    test_seconds=round(test_seconds, 1),
                    touched_tests=touched,
                    note=("edited the tests" if touched else note),
                    tools=telemetry.get("calls_by_tool", {}),
                )
            )
            mark = "pass" if results[-1].passed else "FAIL"
            print(
                f"  {language:8s} {exercise.name:28s} {mark:5s} "
                f"turns={results[-1].turns:3d} ${cost:.4f} {agent_seconds:6.1f}s  {results[-1].note[:60]}"
            )

    write_report(Path(args.out), args, results, spent)


def write_report(path: Path, args, results: list[Outcome], spent: float) -> None:
    if not results:
        print("nothing ran")
        return
    solved = [r for r in results if r.passed]
    by_lang: dict[str, list[Outcome]] = {}
    for r in results:
        by_lang.setdefault(r.language, []).append(r)

    lines = [
        "# Polyglot benchmark",
        "",
        f"Exercises from [Aider-AI/polyglot-benchmark](https://github.com/Aider-AI/polyglot-benchmark),",
        f"the Exercism set. Model `{args.model}`, seed {args.seed}, "
        f"{args.per_language} exercises per language.",
        "",
        "The agent is given the instructions, the stub file name and the test command. "
        "It may not edit the test files; a run that does is scored as a failure "
        "whether or not the suite then passes. Scoring is the exercise's own test "
        "suite, so it is the same number on every run.",
        "",
        f"**Solved {len(solved)} of {len(results)} ({len(solved) / len(results):.0%})** "
        f"for ${spent:.2f} total.",
        "",
        "| language | solved | median turns | median cost | median wall |",
        "|---|---:|---:|---:|---:|",
    ]
    for language, rows in by_lang.items():
        ok = [r for r in rows if r.passed]
        lines.append(
            f"| {language} | {len(ok)}/{len(rows)} ({len(ok) / len(rows):.0%}) "
            f"| {statistics.median(r.turns for r in rows):.0f} "
            f"| ${statistics.median(r.cost_usd for r in rows):.4f} "
            f"| {statistics.median(r.agent_seconds for r in rows):.0f}s |"
        )

    failures = [r for r in results if not r.passed]
    if failures:
        lines += [
            "",
            "## Failures",
            "",
            "| exercise | agent stopped | turns | why |",
            "|---|---|---:|---|",
        ]
        for r in failures:
            lines.append(
                f"| {r.language}/{r.exercise} | {r.agent_outcome} | {r.turns} | {r.note[:90]} |"
            )

    tools: dict[str, int] = {}
    for r in results:
        for name, count in r.tools.items():
            tools[name] = tools.get(name, 0) + int(count)
    if tools:
        lines += [
            "",
            "## Tool calls",
            "",
            " · ".join(f"`{k}` {v}" for k, v in sorted(tools.items(), key=lambda kv: -kv[1])),
        ]

    lines += ["", f"Raw results: `{path.with_suffix('.json').name}`.", ""]
    path.write_text("\n".join(lines))
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "model": args.model,
                "seed": args.seed,
                "spent_usd": round(spent, 4),
                "results": [asdict(r) for r in results],
            },
            indent=1,
        )
    )
    print("\n" + "\n".join(lines[:14]))


if __name__ == "__main__":
    main()
