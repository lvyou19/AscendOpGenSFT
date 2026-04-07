#!/usr/bin/env python3

import argparse
import shlex
import shutil
import subprocess
import sys
import os
from pathlib import Path

codex_cmd = [
    "codex",
    "--sandbox",
    "workspace-write",
    "--ask-for-approval",
    "on-request",
    "执行 AGENT.md 中的完整流程",
]

claude_cmd = [
    "claude",
    "--dangerously-skip-permissions", 
    "--model",
    "kimi",
    "执行 AGENT.md 中的完整流程",
]

# 2. 复制当前环境变量
env = os.environ.copy()
# 3. 添加或更新 IS_SANDBOX 变量
env["IS_SANDBOX"] = "1"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a single torch reference file into agent_workdir/current_task, "
            "run an agent command, then archive the finished task."
        )
    )
    parser.add_argument("reference_py", help="Path to a single torch reference .py file")
    parser.add_argument(
        "task_name",
        nargs="?",
        help="Archive task name. Defaults to the reference filename stem.",
    )
    parser.add_argument(
        "--agent",
        choices=("codex", "claude"),
        help="Use a built-in preset command for codex or claude.",
    )
    parser.add_argument(
        "--agent-cmd",
        nargs=argparse.REMAINDER,
        help="Custom agent command to run. Example: --agent-cmd codex",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # ASCEND_OPGENSFT_WORKDIR配置为agnet的agent_workdir路径
    workdir = Path(os.environ.get("ASCEND_OPGENSFT_WORKDIR", Path(__file__).resolve().parent))
    agent_workdir = Path(os.environ.get("ASCEND_AGENT_WORKDIR", workdir / "agent_workdir"))
    current_task_dir = agent_workdir / "current_task"
    archive_root = workdir / "archive_tasks"

    reference_file = Path(args.reference_py).expanduser().resolve()
    if not reference_file.is_file():
        raise FileNotFoundError(f"Reference file not found: {reference_file}")
    if reference_file.suffix != ".py":
        raise ValueError(f"Reference file must be a .py file: {reference_file}")

    task_name = args.task_name or reference_file.stem
    archive_task_dir = archive_root / task_name
    if archive_task_dir.exists():
        raise FileExistsError(f"Archive target already exists: {archive_task_dir}")

    if args.agent_cmd and args.agent:
        raise ValueError("Use either --agent or --agent-cmd, not both.")

    if args.agent_cmd:
        agent_cmd = list(args.agent_cmd)
        if agent_cmd and agent_cmd[0] == "--":
            agent_cmd = agent_cmd[1:]
    elif args.agent == "codex":
        agent_cmd = codex_cmd.copy()
    elif args.agent == "claude":
        agent_cmd = claude_cmd.copy()
    else:
        raise ValueError("Missing agent command. Use --agent codex/claude or --agent-cmd ...")

    if not agent_cmd:
        raise ValueError("Resolved empty agent command.")

    archive_root.mkdir(parents=True, exist_ok=True)

    if current_task_dir.exists():
        shutil.rmtree(current_task_dir)
    (current_task_dir / "design" / "block_level").mkdir(parents=True, exist_ok=True)
    (current_task_dir / "design" / "tile_level").mkdir(parents=True, exist_ok=True)
    (current_task_dir / "kernel").mkdir(parents=True, exist_ok=True)

    shutil.copy2(reference_file, current_task_dir / "model.py")

    print(f"[1/3] Prepared current task from {reference_file}")
    print(f"       task_name={task_name}")

    print(f"[2/3] Running agent command from {agent_workdir}")
    print(f"       cmd={shlex.join(agent_cmd)}")
    subprocess.run(agent_cmd, cwd=agent_workdir, check=True, env=env)

    print(f"[3/3] Archiving current task to {archive_task_dir}")
    shutil.move(str(current_task_dir), str(archive_task_dir))
    current_task_dir.mkdir(parents=True, exist_ok=True)
    (current_task_dir / "model.py").write_text("", encoding="utf-8")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
