"""CLI entry point: `mini-harness` (interactive) or `mini-harness --task "..."` (one shot)."""

import argparse

from mini_harness.agent import DeepSeekAgent
from mini_harness.config import CONFIG
from mini_harness.tool.box import TOOLS
from mini_harness.tool.tag import OUTCOME

# Exit codes a benchmark driver can branch on.
EXIT = {
    OUTCOME.COMPLETED: 0,
    OUTCOME.ERROR: 1,
    OUTCOME.EXHAUSTED: 3,
    OUTCOME.TIMEOUT: 4,
    OUTCOME.INTERRUPTED: 130,
}


def main(cfg=CONFIG) -> None:
    parser = argparse.ArgumentParser(prog="mini-harness")
    parser.add_argument("--task", default=None, help="run one task unattended and exit")
    parser.add_argument("--telemetry-out", default=None, help="write the run's JSON telemetry here")
    args = parser.parse_args()

    agent = DeepSeekAgent(TOOLS, cfg=cfg)
    print(f"[mini-harness]: profile={cfg.profile} workspace={cfg.work_space} "
          f"guard={cfg.guard_read}/{cfg.guard_write} turns={cfg.max_turns_main} wall={cfg.wall_budget}")

    if not args.task:
        agent.run(cfg=cfg)
        return

    result = agent.run_task(args.task, cfg=cfg)
    agent.dump_run(result, args.task, args.telemetry_out, cfg=cfg)
    raise SystemExit(EXIT.get(result.outcome, 1))


if __name__ == "__main__":
    main()
