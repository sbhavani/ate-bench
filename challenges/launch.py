import argparse
import json
import os
import re
import select
import secrets
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
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
        instruction_prefix_files: list[str],
        run_label: str | None,
        codex_bypass_approvals: bool,
        codex_sandbox: str,
        agent_container_image: str | None,
        agent_container_user: str | None,
        agent_container_privileged: bool,
        success_artifact: str | None,
        success_artifact_contains: str | None,
        stop_after_success_artifact: bool,
        success_artifact_grace_sec: float,
        skip_agent: bool,
        keep_workspace: bool,
    ):
        self.framework, self.challenge, self.agent, self.model = framework, challenge, agent, model
        self.overlays = overlays
        self.instruction_prefix_files = instruction_prefix_files
        self.run_label = self.sanitize_run_label(run_label)
        self.codex_bypass_approvals = codex_bypass_approvals
        self.codex_sandbox = codex_sandbox
        self.agent_container_image = agent_container_image
        self.agent_container_user = agent_container_user
        self.agent_container_privileged = agent_container_privileged
        self.success_artifact = success_artifact
        self.success_artifact_contains = success_artifact_contains
        self.stop_after_success_artifact = stop_after_success_artifact
        self.success_artifact_grace_sec = success_artifact_grace_sec
        self.skip_agent = skip_agent
        self.keep_workspace = keep_workspace
        uuid_parts = [framework]
        if self.run_label:
            uuid_parts.append(self.run_label)
        uuid_parts.append(secrets.token_hex(3))
        self.uuid = "-".join(uuid_parts)
        self.workspace = Path(WORKSPACE, challenge, self.uuid)
        self.workspace.mkdir(parents=True, exist_ok=False)
        self.launch_started_monotonic = time.monotonic()
        self.launch_started_epoch = time.time()
        self.launch_started_at = self.utc_now()

    @staticmethod
    def utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def iso_from_epoch(epoch: float) -> str:
        return datetime.fromtimestamp(epoch, timezone.utc).isoformat()

    def write_run_metrics(self, updates: dict):
        metrics = {}
        metrics_path = Path(self.workspace, "artifacts", "run-metrics.json")
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
            except json.JSONDecodeError:
                metrics = {}
        metrics.update(updates)
        metrics["updated_at"] = self.utc_now()
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    def prepare(self):
        # Materialize the workspace: the per-challenge script clones, patches, and builds the framework.
        Path(self.workspace, "artifacts").mkdir()
        prepare_started = time.monotonic()
        self.write_run_metrics(
            {
                "launch_started_at": self.launch_started_at,
                "prepare_started_at": self.utc_now(),
            }
        )
        prepare = Path(self.challenge, "prepare", "%s.sh" % self.framework).as_posix()
        subprocess.run(["bash", prepare, self.workspace.as_posix()], check=True)
        self.apply_overlays()
        self.write_run_metrics(
            {
                "prepare_finished_at": self.utc_now(),
                "prepare_elapsed_sec": round(time.monotonic() - prepare_started, 3),
            }
        )

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
        attempt_started_monotonic = time.monotonic()
        attempt_started_epoch = time.time()
        self.write_run_metrics(
            {
                "attempt_started_at": self.utc_now(),
                "agent": self.agent,
            }
        )
        instruction = Path(self.challenge, "instruction.md").read_text()
        instruction = instruction.format(framework=self.framework)
        if self.instruction_prefix_files:
            prefixes = [Path(prefix_file).read_text().rstrip() for prefix_file in self.instruction_prefix_files]
            instruction = "\n\n".join(prefixes + [instruction])
        phase_log = Path(self.workspace, "artifacts", "install-phases.jsonl")
        os.environ["ATE_INSTALL_PHASE_LOG"] = phase_log.as_posix()
        metadata = {
            "framework": self.framework,
            "challenge": self.challenge,
            "agent": self.agent,
            "model": self.model,
            "run_label": self.run_label,
            "overlays": self.overlays,
            "instruction_prefix_files": self.instruction_prefix_files,
            "codex_bypass_approvals": self.codex_bypass_approvals,
            "codex_sandbox": self.codex_sandbox,
            "agent_container_image": self.agent_container_image,
            "agent_container_user": self.agent_container_user,
            "agent_container_privileged": self.agent_container_privileged,
            "success_artifact": self.success_artifact,
            "success_artifact_contains": self.success_artifact_contains,
            "stop_after_success_artifact": self.stop_after_success_artifact,
            "success_artifact_grace_sec": self.success_artifact_grace_sec,
            "workspace": self.workspace.as_posix(),
            "install_phase_log": "artifacts/install-phases.jsonl",
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
            self.write_run_metrics(
                {
                    "attempt_finished_at": self.utc_now(),
                    "attempt_elapsed_sec": round(time.monotonic() - attempt_started_monotonic, 3),
                    "agent_skipped": True,
                }
            )
            return
        try:
            self.run_agent(args, attempt_started_monotonic, attempt_started_epoch)
        finally:
            self.write_run_metrics(
                {
                    "attempt_finished_at": self.utc_now(),
                    "attempt_elapsed_sec": round(time.monotonic() - attempt_started_monotonic, 3),
                }
            )

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
        policy_rule = self.write_codex_exec_policy_rules()
        last_message = Path(self.workspace, "artifacts", "codex-last-message.txt")
        args = ["codex", "exec", "--json"]
        args.extend(["-c", 'shell_environment_policy.inherit="all"'])
        args.extend(["--sandbox", self.codex_sandbox])
        args.extend(["-C", self.workspace.as_posix()])
        args.extend(["-o", last_message.as_posix()])
        if self.model:
            args.extend(["--model", self.model])
        # ATE-Bench workspaces are disposable sandboxes. Match Claude Code's
        # permission-skipping behavior so environment setup and profiling tasks
        # can run without interactive approval prompts. Managed enterprise
        # Codex policies may disallow this flag; callers can opt out and rely
        # on the configured sandbox plus local trust rules instead.
        if self.codex_bypass_approvals:
            args.append("--dangerously-bypass-approvals-and-sandbox")
        args.append(instruction)
        if self.agent_container_image:
            args = self.containerized_agent_args(args, policy_rule)
        return args

    def write_codex_exec_policy_rules(self):
        rules_dir = Path(self.workspace, ".codex", "rules")
        rules_dir.mkdir(parents=True, exist_ok=True)
        policy_rule = Path(rules_dir, "default.rules")
        policy_rule.write_text(
            "\n".join(
                [
                    'prefix_rule(pattern=["awk"], decision="allow")',
                    'prefix_rule(pattern=["cat"], decision="allow")',
                    'prefix_rule(pattern=["chmod"], decision="allow")',
                    'prefix_rule(pattern=["cmake"], decision="allow")',
                    'prefix_rule(pattern=["cp"], decision="allow")',
                    'prefix_rule(pattern=["df"], decision="allow")',
                    'prefix_rule(pattern=["dirname"], decision="allow")',
                    'prefix_rule(pattern=["du"], decision="allow")',
                    'prefix_rule(pattern=["env"], decision="allow")',
                    'prefix_rule(pattern=["find"], decision="allow")',
                    'prefix_rule(pattern=["git"], decision="allow")',
                    'prefix_rule(pattern=["grep"], decision="allow")',
                    'prefix_rule(pattern=["head"], decision="allow")',
                    'prefix_rule(pattern=["ldconfig"], decision="allow")',
                    'prefix_rule(pattern=["ln"], decision="allow")',
                    'prefix_rule(pattern=["ls"], decision="allow")',
                    'prefix_rule(pattern=["mkdir"], decision="allow")',
                    'prefix_rule(pattern=["ninja"], decision="allow")',
                    'prefix_rule(pattern=["ninja-build"], decision="allow")',
                    'prefix_rule(pattern=["nvidia-smi"], decision="allow")',
                    'prefix_rule(pattern=["nvcc"], decision="allow")',
                    'prefix_rule(pattern=["pip"], decision="allow")',
                    'prefix_rule(pattern=["printenv"], decision="allow")',
                    'prefix_rule(pattern=["printf"], decision="allow")',
                    'prefix_rule(pattern=["ps"], decision="allow")',
                    'prefix_rule(pattern=["pwd"], decision="allow")',
                    'prefix_rule(pattern=["python"], decision="allow")',
                    'prefix_rule(pattern=["python3"], decision="allow")',
                    'prefix_rule(pattern=["python3.12"], decision="allow")',
                    'prefix_rule(pattern=["realpath"], decision="allow")',
                    'prefix_rule(pattern=["rm"], decision="allow")',
                    'prefix_rule(pattern=["rmdir"], decision="allow")',
                    'prefix_rule(pattern=["rg"], decision="allow")',
                    'prefix_rule(pattern=["sed"], decision="allow")',
                    'prefix_rule(pattern=["tail"], decision="allow")',
                    'prefix_rule(pattern=["tee"], decision="allow")',
                    'prefix_rule(pattern=["test"], decision="allow")',
                    'prefix_rule(pattern=["touch"], decision="allow")',
                    'prefix_rule(pattern=["uv"], decision="allow")',
                    'prefix_rule(pattern=["uvx"], decision="allow")',
                    'prefix_rule(pattern=["which"], decision="allow")',
                    'prefix_rule(pattern=[".venv/bin/python"], decision="allow")',
                    'prefix_rule(pattern=["./.venv/bin/python"], decision="allow")',
                    'prefix_rule(pattern=["Megatron-LM/.venv/bin/python"], decision="allow")',
                    'prefix_rule(pattern=["./Megatron-LM/.venv/bin/python"], decision="allow")',
                    'prefix_rule(pattern=["/home/shadeform/.local/bin/uv"], decision="allow")',
                    'prefix_rule(pattern=["/usr/bin/env"], decision="allow")',
                    'prefix_rule(pattern=["/usr/bin/git"], decision="allow")',
                    'prefix_rule(pattern=["/usr/bin/nvidia-smi"], decision="allow")',
                    'prefix_rule(pattern=["/usr/bin/python3"], decision="allow")',
                    'prefix_rule(pattern=["/usr/bin/python3.12"], decision="allow")',
                    'prefix_rule(pattern=["/usr/local/cuda/bin/nvcc"], decision="allow")',
                    'prefix_rule(pattern=["/usr/bin/bash"], decision="allow")',
                    'prefix_rule(pattern=["/usr/bin/bash", "-c"], decision="allow")',
                    'prefix_rule(pattern=["/usr/bin/bash", "-lc"], decision="allow")',
                    'prefix_rule(pattern=["/bin/bash"], decision="allow")',
                    'prefix_rule(pattern=["/bin/bash", "-c"], decision="allow")',
                    'prefix_rule(pattern=["/bin/bash", "-lc"], decision="allow")',
                    'prefix_rule(pattern=["bash"], decision="allow")',
                    'prefix_rule(pattern=["bash", "-c"], decision="allow")',
                    'prefix_rule(pattern=["bash", "-lc"], decision="allow")',
                    "",
                ]
            )
        )
        return policy_rule

    def containerized_agent_args(self, inner_args: list[str], policy_rule: Path | None):
        home = Path.home()
        env_names = [
            "CUDA_HOME",
            "CUDA_PATH",
            "HF_HOME",
            "HF_TOKEN_PATH",
            "LD_LIBRARY_PATH",
            "ATE_INSTALL_PHASE_LOG",
            "PIP_CACHE_DIR",
            "PYTHONUNBUFFERED",
            "UV_CACHE_DIR",
        ]
        args = [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--network",
            "host",
            "-e",
            "HOME=%s" % home,
            "-e",
            "PATH=%s" % os.environ.get("PATH", ""),
        ]
        if self.agent_container_privileged:
            args.append("--privileged")
        if self.agent_container_user:
            args.extend(["--user", self.agent_container_user])
        for name in env_names:
            if name in os.environ:
                args.extend(["-e", "%s=%s" % (name, os.environ[name])])
        for path in [
            Path.cwd(),
            Path(os.environ.get("UV_CACHE_DIR", "")).parent if os.environ.get("UV_CACHE_DIR") else None,
            home / "node20",
            home / ".codex",
            home / ".local",
        ]:
            if path and path.exists():
                args.extend(["-v", "%s:%s" % (path, path)])
        if policy_rule and policy_rule.exists():
            container_rules = Path(home, ".codex", "rules")
            args.extend(["-v", "%s:%s:ro" % (policy_rule.parent, container_rules)])
        args.extend(["-w", self.workspace.as_posix(), self.agent_container_image])
        args.extend(inner_args)
        return args

    def run_agent(self, args, attempt_started_monotonic: float, attempt_started_epoch: float):
        events = Path(self.workspace, "artifacts", "%s-events.jsonl" % self.agent)
        timed_events = Path(self.workspace, "artifacts", "%s-events-timed.jsonl" % self.agent)
        self.write_agent_command(args)
        agent_started_monotonic = time.monotonic()
        agent_started_epoch = time.time()
        self.write_run_metrics(
            {
                "agent_started_at": self.utc_now(),
            }
        )
        with events.open("wb") as log, timed_events.open("w") as timed_log:
            proc = subprocess.Popen(args, cwd=self.workspace, stdout=subprocess.PIPE)
            assert proc.stdout is not None
            stopped_after_success = False
            success_metric_written = False
            success_ready_monotonic = None
            while True:
                ready, _, _ = select.select([proc.stdout], [], [], 1.0)
                if ready:
                    line = proc.stdout.readline()
                    if line:
                        sys.stdout.buffer.write(line)
                        sys.stdout.buffer.flush()
                        log.write(line)
                        log.flush()
                        self.write_timed_event(timed_log, line, agent_started_epoch)
                    elif proc.poll() is not None:
                        break
                elif proc.poll() is not None:
                    break

                success_metrics = self.success_artifact_metrics(
                    attempt_started_epoch, agent_started_epoch
                )
                success_ready = success_metrics.get("ready")
                if success_ready and success_ready_monotonic is None:
                    success_ready_monotonic = time.monotonic()
                if success_ready and not success_metric_written:
                    self.write_run_metrics(
                        {
                            "success_artifact": success_metrics,
                            "install_smoke": self.install_smoke_metrics(
                                attempt_started_epoch, agent_started_epoch
                            ),
                        }
                    )
                    success_metric_written = True
                if self.stop_after_success_artifact and success_ready:
                    grace_elapsed = 0.0
                    if success_ready_monotonic is not None:
                        grace_elapsed = time.monotonic() - success_ready_monotonic
                    if (
                        self.success_artifact_grace_sec > 0
                        and grace_elapsed < self.success_artifact_grace_sec
                    ):
                        continue
                    stopped_after_success = True
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    break
            returncode = proc.wait()
        phase_metrics = self.write_install_phase_metrics()
        self.write_run_metrics(
            {
                "agent_finished_at": self.utc_now(),
                "agent_elapsed_sec": round(time.monotonic() - agent_started_monotonic, 3),
                "attempt_elapsed_sec": round(time.monotonic() - attempt_started_monotonic, 3),
                "agent_returncode": returncode,
                "stopped_after_success_artifact": stopped_after_success,
                "success_artifact": self.success_artifact_metrics(
                    attempt_started_epoch, agent_started_epoch
                ),
                "install_smoke": self.install_smoke_metrics(attempt_started_epoch, agent_started_epoch),
                "install_phases": phase_metrics,
            }
        )
        if returncode and not stopped_after_success:
            raise subprocess.CalledProcessError(returncode, args)

    def write_timed_event(self, timed_log, line: bytes, agent_started_epoch: float):
        received_epoch = time.time()
        text = line.decode("utf-8", errors="replace").rstrip("\n")
        payload = {
            "received_at": self.iso_from_epoch(received_epoch),
            "received_epoch": round(received_epoch, 6),
            "since_agent_start_sec": round(max(0.0, received_epoch - agent_started_epoch), 3),
        }
        try:
            payload["event"] = json.loads(text)
        except json.JSONDecodeError:
            payload["raw"] = text
        timed_log.write(json.dumps(payload, sort_keys=True) + "\n")
        timed_log.flush()

    def write_install_phase_metrics(self):
        timed_events = Path(self.workspace, "artifacts", "%s-events-timed.jsonl" % self.agent)
        metrics_path = Path(self.workspace, "artifacts", "install-phase-metrics.json")
        timeline_path = Path(self.workspace, "artifacts", "command-timeline.json")
        if not timed_events.exists():
            metrics = {"available": False, "reason": "timed event stream not found"}
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
            return metrics

        commands_by_id: dict[str, dict] = {}
        with timed_events.open() as f:
            for raw in f:
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                event = payload.get("event")
                if not isinstance(event, dict):
                    continue
                item = event.get("item")
                if not isinstance(item, dict) or item.get("type") != "command_execution":
                    continue
                command_id = str(item.get("id") or len(commands_by_id))
                command = item.get("command") or ""
                record = commands_by_id.setdefault(
                    command_id,
                    {
                        "id": command_id,
                        "command": command,
                    },
                )
                if command and not record.get("command"):
                    record["command"] = command
                if event.get("type") == "item.started":
                    record["started_at"] = payload.get("received_at")
                    record["started_epoch"] = payload.get("received_epoch")
                    record["started_since_agent_sec"] = payload.get("since_agent_start_sec")
                elif event.get("type") == "item.completed":
                    record["finished_at"] = payload.get("received_at")
                    record["finished_epoch"] = payload.get("received_epoch")
                    record["finished_since_agent_sec"] = payload.get("since_agent_start_sec")
                    for field in ("exit_code", "status"):
                        if field in item:
                            record[field] = item[field]

        timeline = []
        for record in commands_by_id.values():
            started = record.get("started_epoch")
            finished = record.get("finished_epoch")
            if isinstance(started, (int, float)) and isinstance(finished, (int, float)):
                record["elapsed_sec"] = round(max(0.0, finished - started), 3)
            record["phase"] = self.classify_install_command(record.get("command", ""))
            timeline.append(record)
        timeline.sort(key=lambda item: item.get("started_epoch") or item.get("finished_epoch") or 0)

        command_phases: dict[str, dict] = {}
        for record in timeline:
            phase = record["phase"]
            phase_metrics = command_phases.setdefault(
                phase,
                {
                    "commands": 0,
                    "elapsed_sec": 0.0,
                    "first_started_since_agent_sec": None,
                    "last_finished_since_agent_sec": None,
                },
            )
            phase_metrics["commands"] += 1
            elapsed = record.get("elapsed_sec")
            if isinstance(elapsed, (int, float)):
                phase_metrics["elapsed_sec"] = round(phase_metrics["elapsed_sec"] + elapsed, 3)
            started_since = record.get("started_since_agent_sec")
            finished_since = record.get("finished_since_agent_sec")
            if isinstance(started_since, (int, float)) and (
                phase_metrics["first_started_since_agent_sec"] is None
                or started_since < phase_metrics["first_started_since_agent_sec"]
            ):
                phase_metrics["first_started_since_agent_sec"] = started_since
            if isinstance(finished_since, (int, float)) and (
                phase_metrics["last_finished_since_agent_sec"] is None
                or finished_since > phase_metrics["last_finished_since_agent_sec"]
            ):
                phase_metrics["last_finished_since_agent_sec"] = finished_since

        explicit_phases = self.read_explicit_phase_metrics()
        if explicit_phases:
            phases = explicit_phases
            phase_source = "install-phases.jsonl"
        else:
            phases = command_phases
            phase_source = "%s-events-timed.jsonl" % self.agent

        metrics = {
            "available": True,
            "source": phase_source,
            "command_timeline_source": "%s-events-timed.jsonl" % self.agent,
            "timeline": "artifacts/command-timeline.json",
            "commands": len(timeline),
            "phases": phases,
            "command_phases": command_phases,
        }
        timeline_path.write_text(json.dumps(timeline, indent=2, sort_keys=True) + "\n")
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        return metrics

    def read_explicit_phase_metrics(self):
        phase_log = Path(self.workspace, "artifacts", "install-phases.jsonl")
        if not phase_log.exists():
            return {}

        active: dict[str, list[dict]] = {}
        phases: dict[str, dict] = {}
        with phase_log.open() as f:
            for raw in f:
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                phase = event.get("phase")
                event_type = event.get("event")
                epoch = event.get("epoch")
                since_agent = event.get("since_agent_start_sec")
                if not isinstance(phase, str) or not isinstance(event_type, str):
                    continue
                if event_type == "start":
                    active.setdefault(phase, []).append(event)
                    metrics = phases.setdefault(
                        phase,
                        {
                            "commands": None,
                            "elapsed_sec": 0.0,
                            "first_started_since_agent_sec": None,
                            "last_finished_since_agent_sec": None,
                        },
                    )
                    if isinstance(since_agent, (int, float)) and (
                        metrics["first_started_since_agent_sec"] is None
                        or since_agent < metrics["first_started_since_agent_sec"]
                    ):
                        metrics["first_started_since_agent_sec"] = since_agent
                elif event_type == "end":
                    starts = active.get(phase) or []
                    start = starts.pop() if starts else {}
                    start_epoch = start.get("epoch")
                    elapsed = event.get("elapsed_sec")
                    if (
                        not isinstance(elapsed, (int, float))
                        and isinstance(start_epoch, (int, float))
                        and isinstance(epoch, (int, float))
                    ):
                        elapsed = max(0.0, epoch - start_epoch)
                    metrics = phases.setdefault(
                        phase,
                        {
                            "commands": None,
                            "elapsed_sec": 0.0,
                            "first_started_since_agent_sec": start.get("since_agent_start_sec"),
                            "last_finished_since_agent_sec": None,
                        },
                    )
                    if isinstance(elapsed, (int, float)):
                        metrics["elapsed_sec"] = round(metrics["elapsed_sec"] + elapsed, 3)
                    if isinstance(since_agent, (int, float)) and (
                        metrics["last_finished_since_agent_sec"] is None
                        or since_agent > metrics["last_finished_since_agent_sec"]
                    ):
                        metrics["last_finished_since_agent_sec"] = since_agent
        return phases

    @staticmethod
    def classify_install_command(command: str) -> str:
        command_l = command.lower()
        if "install_te_pypi.sh" in command_l:
            return "megatron+te-install"
        if "pip install" in command_l and (
            "torch==" in command_l
            or "--torch-backend" in command_l
            or "download.pytorch.org" in command_l
        ):
            return "torch-bootstrap"
        if "uv venv" in command_l or "python -m venv" in command_l:
            return "torch-bootstrap"
        if (
            "install-smoke.log" in command_l
            or "install smoke: ok" in command_l
            or "transformer_engine" in command_l
            or "transformer-engine" in command_l
            or "pip install" in command_l
            and (
                " -e " in command_l
                or "megatron-core" in command_l
                or "megatron-lm" in command_l
                or " .[" in command_l
                or " -e ." in command_l
            )
        ):
            return "megatron+te-install"
        if any(
            marker in command_l
            for marker in (
                "rg ",
                "sed ",
                "cat ",
                "git status",
                "git log",
                "uv --version",
                "python",
                "nvidia-smi",
                "nvcc --version",
                "ps ",
                "du ",
            )
        ):
            return "inspect"
        return "other"

    def install_smoke_metrics(self, attempt_started_epoch: float, agent_started_epoch: float | None = None):
        smoke = Path(self.workspace, "artifacts", "install-smoke.log")
        if not smoke.exists():
            return {"exists": False}
        try:
            text = smoke.read_text(errors="replace")
        except OSError:
            text = ""
        mtime = smoke.stat().st_mtime
        metrics = {
            "exists": True,
            "path": "artifacts/install-smoke.log",
            "mtime_at": self.iso_from_epoch(mtime),
            "since_attempt_start_sec": round(max(0.0, mtime - attempt_started_epoch), 3),
            "contains_install_smoke_ok": "install smoke: ok" in text,
        }
        if agent_started_epoch is not None:
            metrics["since_agent_start_sec"] = round(max(0.0, mtime - agent_started_epoch), 3)
        return metrics

    def success_artifact_metrics(
        self, attempt_started_epoch: float, agent_started_epoch: float | None = None
    ):
        if not self.success_artifact:
            return {"configured": False, "ready": False}
        artifact = Path(self.workspace, self.success_artifact)
        metrics = {
            "configured": True,
            "ready": False,
            "exists": artifact.exists(),
            "path": self.success_artifact,
        }
        if not artifact.exists():
            return metrics
        try:
            text = artifact.read_text(errors="replace")
        except OSError:
            text = ""
        mtime = artifact.stat().st_mtime
        contains = True
        if self.success_artifact_contains:
            contains = self.success_artifact_contains in text
        metrics.update(
            {
                "ready": contains,
                "contains_expected_text": contains,
                "mtime_at": self.iso_from_epoch(mtime),
                "since_attempt_start_sec": round(max(0.0, mtime - attempt_started_epoch), 3),
            }
        )
        if agent_started_epoch is not None:
            metrics["since_agent_start_sec"] = round(max(0.0, mtime - agent_started_epoch), 3)
        return metrics

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
            self.write_run_metrics(
                {
                    "benchmark_finished_at": self.utc_now(),
                    "benchmark_elapsed_sec": round(time.monotonic() - self.launch_started_monotonic, 3),
                }
            )
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
        "--codex-bypass-approvals",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("ATE_CODEX_BYPASS_APPROVALS", "1").lower()
        not in {"0", "false", "no"},
        help="Use Codex's approval/sandbox bypass flag for disposable ATE workspaces.",
    )
    p.add_argument(
        "--agent-container-image",
        type=str,
        default=None,
        help="Run the agent inside this Docker image, mounting the ATE workspace and local Codex/uv tools.",
    )
    p.add_argument(
        "--agent-container-user",
        type=str,
        default="%s:%s" % (os.getuid(), os.getgid()),
        help="User passed to Docker for the agent container; use `root` if nested sandbox namespaces fail.",
    )
    p.add_argument(
        "--agent-container-privileged",
        action="store_true",
        help="Run the agent container with Docker --privileged when nested sandbox namespaces require it.",
    )
    p.add_argument(
        "--codex-sandbox",
        type=str,
        choices=["read-only", "workspace-write", "danger-full-access"],
        default="workspace-write",
        help="Sandbox mode passed to `codex exec`; use danger-full-access only inside an external sandbox.",
    )
    p.add_argument(
        "--success-artifact",
        type=str,
        default=None,
        help="Relative workspace path whose creation marks task success for timing metrics.",
    )
    p.add_argument(
        "--success-artifact-contains",
        type=str,
        default=None,
        help="Optional text that must appear in --success-artifact before it is considered ready.",
    )
    p.add_argument(
        "--stop-after-success-artifact",
        action="store_true",
        help="Terminate the agent after the success artifact is ready, then capture artifacts.",
    )
    p.add_argument(
        "--success-artifact-grace-sec",
        type=float,
        default=0.0,
        help=(
            "After --success-artifact is ready, wait this many seconds for the agent to exit "
            "naturally before terminating it. Useful for preserving final usage metrics."
        ),
    )
    p.add_argument(
        "--overlay",
        action="append",
        default=[],
        metavar="SOURCE:DESTINATION",
        help="Copy SOURCE into the prepared workspace at DESTINATION before the agent runs. May repeat.",
    )
    p.add_argument(
        "--instruction-prefix-file",
        action="append",
        default=[],
        type=str,
        help="Prepend this file's contents to the challenge instruction before running the agent. May repeat.",
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
        a.codex_bypass_approvals,
        a.codex_sandbox,
        a.agent_container_image,
        a.agent_container_user,
        a.agent_container_privileged,
        a.success_artifact,
        a.success_artifact_contains,
        a.stop_after_success_artifact,
        a.success_artifact_grace_sec,
        a.skip_agent,
        a.keep_workspace,
    ).launch()
