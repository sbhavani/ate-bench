import argparse
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path("workspace").resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)
UV_CACHE_DIR = Path("workspace/uv-cache").resolve()
UV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
HF_HOME = Path("workspace/hf-home").resolve()
HF_HOME.mkdir(parents=True, exist_ok=True)
HF_TOKEN_PATH = Path(Path.home(), ".cache/huggingface/token")

os.environ.setdefault("UV_CACHE_DIR", UV_CACHE_DIR.as_posix())
os.environ.setdefault("HF_HOME", HF_HOME.as_posix())
os.environ.setdefault("HF_TOKEN_PATH", HF_TOKEN_PATH.as_posix())
os.environ.setdefault("PYTHONUNBUFFERED", "1")

for tool in ("uv",):
    if shutil.which(tool) is None:
        raise SystemExit("required tool not on PATH: %s" % tool)


class Runner:

    def __init__(
        self,
        framework: str,
        challenge: str,
        agent: str,
        model: str | None,
        overlays: list[str],
        instruction_prefix_file: str | None,
        run_label: str | None,
        skip_agent: bool,
        keep_workspace: bool,
    ):
        self.framework, self.challenge, self.agent, self.model = framework, challenge, agent, model
        self.overlays = overlays
        self.instruction_prefix_file = instruction_prefix_file
        self.run_label = self.sanitize_run_label(run_label)
        self.skip_agent = skip_agent
        self.keep_workspace = keep_workspace
        uuid_parts = [framework]
        if self.run_label:
            uuid_parts.append(self.run_label)
        uuid_parts.append(secrets.token_hex(3))
        self.uuid = "-".join(uuid_parts)
        self.workspace = Path(WORKSPACE, challenge, self.uuid)
        self.workspace.mkdir(parents=True, exist_ok=False)

    def prepare(self):
        # Materialize the workspace: the per-challenge script clones, patches, and builds the framework.
        Path(self.workspace, "artifacts").mkdir()
        prepare = Path(self.challenge, "prepare", "%s.sh" % self.framework).as_posix()
        subprocess.run(["bash", prepare, self.workspace.as_posix()], check=True)
        self.apply_overlays()

    def apply_overlays(self):
        for overlay in self.overlays:
            source, destination = self.parse_overlay(overlay)
            if not source.exists():
                raise FileNotFoundError("overlay source not found: %s" % source)
            target = Path(self.workspace, destination)
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True, ignore=self.overlay_ignore)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

    @staticmethod
    def parse_overlay(overlay: str):
        if ":" not in overlay:
            raise ValueError("overlay must use SOURCE:DESTINATION syntax: %s" % overlay)
        source, destination = overlay.split(":", 1)
        if not source or not destination:
            raise ValueError("overlay must use SOURCE:DESTINATION syntax: %s" % overlay)
        return Path(source).expanduser().resolve(), destination

    @staticmethod
    def sanitize_run_label(run_label: str | None):
        if not run_label:
            return None
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_label.strip()).strip("-")
        if not cleaned:
            raise ValueError("run label must contain at least one alphanumeric, dot, underscore, or dash")
        return cleaned

    @staticmethod
    def overlay_ignore(directory, names):
        ignored = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
        if Path(directory).name == "Megatron-LM":
            ignored.update({"outputs", "workspace", "snapshots"})
        return ignored.intersection(names)

    def attempt(self):
        # Run the agent in the workspace; stream its JSON events to stdout for progress.
        instruction = Path(self.challenge, "instruction.md").read_text()
        instruction = instruction.format(framework=self.framework)
        if self.instruction_prefix_file:
            prefix = Path(self.instruction_prefix_file).read_text()
            instruction = prefix.rstrip() + "\n\n" + instruction
        metadata = {
            "framework": self.framework,
            "challenge": self.challenge,
            "agent": self.agent,
            "model": self.model,
            "run_label": self.run_label,
            "overlays": self.overlays,
            "instruction_prefix_file": self.instruction_prefix_file,
            "workspace": self.workspace.as_posix(),
        }
        Path(self.workspace, "artifacts", "run-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        Path(self.workspace, "artifacts", "%s-instruction.md" % self.agent).write_text(instruction)
        if self.agent == "claude":
            args = self.claude_args(instruction)
        elif self.agent == "codex":
            args = self.codex_args(instruction)
        else:
            raise ValueError("unsupported agent: %s" % self.agent)
        if self.skip_agent:
            self.write_agent_command(args)
            return
        self.run_agent(args)

    def claude_args(self, instruction: str):
        if not self.skip_agent and shutil.which("claude") is None:
            raise SystemExit("required tool not on PATH: claude")
        args = ["claude", "--print"]
        args.extend(["--model", self.model or "claude-opus-4-7", "--effort", "xhigh"])
        args.extend(["--output-format", "stream-json", "--include-partial-messages"])
        # question-and-answer challenges are read-only: the agent investigates the code, never edits it.
        # Keep --disallowedTools ahead of other flags so its variadic value never swallows the instruction.
        if "question-and-answer" in self.challenge:
            args.extend(["--disallowedTools", "Edit,Write,NotebookEdit"])
        args.extend(["--dangerously-skip-permissions", "--verbose"])
        args.append(instruction)
        return args

    def codex_args(self, instruction: str):
        if not self.skip_agent and shutil.which("codex") is None:
            raise SystemExit("required tool not on PATH: codex")
        last_message = Path(self.workspace, "artifacts", "codex-last-message.txt")
        args = ["codex", "exec", "--json"]
        args.extend(["-C", self.workspace.as_posix()])
        args.extend(["-o", last_message.as_posix()])
        if self.model:
            args.extend(["--model", self.model])
        # ATE-Bench workspaces are disposable sandboxes. Match Claude Code's
        # permission-skipping behavior so environment setup and profiling tasks
        # can run without interactive approval prompts.
        args.append("--dangerously-bypass-approvals-and-sandbox")
        args.append(instruction)
        return args

    def run_agent(self, args):
        events = Path(self.workspace, "artifacts", "%s-events.jsonl" % self.agent)
        self.write_agent_command(args)
        with events.open("wb") as log:
            proc = subprocess.Popen(args, cwd=self.workspace, stdout=subprocess.PIPE)
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
                log.write(line)
                log.flush()
            returncode = proc.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, args)

    def write_agent_command(self, args):
        command = Path(self.workspace, "artifacts", "%s-command.txt" % self.agent)
        command.write_text(" ".join(shlex.quote(str(arg)) for arg in args) + "\n")

    def capture(self):
        snapshot = Path("snapshots", self.challenge, self.uuid)
        snapshot.mkdir(parents=True, exist_ok=True)
        patches = Path(snapshot, "patches")
        patches.mkdir(parents=True, exist_ok=True)
        # Capture a patch for each modified codebase; skip the ones the agent left untouched.
        for codebase in sorted(git.parent for git in self.workspace.glob("*/.git")):
            subprocess.run(["git", "add", "-A"], cwd=codebase, check=True)
            if subprocess.run(["git", "diff", "--cached", "--quiet", "main"], cwd=codebase).returncode == 0:
                continue
            with Path(patches, "%s.patch" % codebase.name).open("wb") as patch:
                subprocess.run(["git", "diff", "--cached", "--binary", "main"], cwd=codebase, stdout=patch, check=True)
        # Capture the artifacts and any agent-specific session directory.
        shutil.copytree(Path(self.workspace, "artifacts"), Path(snapshot, "artifacts"), dirs_exist_ok=True)
        if self.agent == "claude":
            project = Path(Path.home(), ".claude/projects", self.workspace.as_posix().replace("/", "-"))
            if project.exists():
                shutil.copytree(project, Path(snapshot, "claude-session"), dirs_exist_ok=True)

    def cleanup(self):
        # Remove the workspace and any claude code session.
        if self.keep_workspace:
            return
        project = Path(Path.home(), ".claude/projects", self.workspace.as_posix().replace("/", "-"))
        shutil.rmtree(self.workspace, ignore_errors=True)
        shutil.rmtree(project, ignore_errors=True)

    def launch(self):
        self.prepare()
        try:
            self.attempt()
        finally:
            # If the attempt failed, we still capture. If the capture failed, we never cleanup.
            # This allows manual inspection over failures at different points of the execution.
            self.capture()
            self.cleanup()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("framework", type=str, choices=["torchtitan", "pith-train", "Megatron-LM"])
    p.add_argument("challenge", type=lambda s: s if Path(s).is_dir() else p.error("%s is not a valid task" % s))
    p.add_argument("--agent", type=str, choices=["claude", "codex"], default="claude")
    p.add_argument("--model", type=str, default=None, help="Agent model override")
    p.add_argument(
        "--overlay",
        action="append",
        default=[],
        metavar="SOURCE:DESTINATION",
        help="Copy SOURCE into the prepared workspace at DESTINATION before the agent runs. May repeat.",
    )
    p.add_argument(
        "--instruction-prefix-file",
        type=str,
        default=None,
        help="Prepend this file's contents to the challenge instruction before running the agent.",
    )
    p.add_argument("--run-label", type=str, default=None, help="Label this run in the workspace/snapshot id")
    p.add_argument("--skip-agent", action="store_true", help="Prepare, overlay, and capture without invoking the agent")
    p.add_argument("--keep-workspace", action="store_true", help="Do not delete the prepared workspace after capture")
    a = p.parse_args()
    Runner(
        a.framework,
        a.challenge,
        a.agent,
        a.model,
        a.overlay,
        a.instruction_prefix_file,
        a.run_label,
        a.skip_agent,
        a.keep_workspace,
    ).launch()
