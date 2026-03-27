"""
Automated Research Agent
========================
An end-to-end research automation system based on the `flow.md` workflow.
Supports two AI backends: Claude Code and OpenAI Codex.

Architecture:
    Orchestrator (this script) ──calls──▶ AI backend (execute steps)
                               ──calls──▶ AI backend (evaluate outputs) ──decide──▶ Orchestrator

Supported AI backends:
    - claude : Claude Code CLI  (npm install -g @anthropic-ai/claude-code)
    - codex  : OpenAI Codex CLI (npm install -g @openai/codex)

Usage:
    python run_research.py [--backend claude|codex] [--start-step N] [--end-step N] [--work-dir PATH]
"""

import json
import logging
import argparse
import subprocess
import os
import sys
import shutil
import re
from pathlib import Path
from datetime import datetime


# ============================================================
# Configuration
# ============================================================

class Config:
    """All configurable parameters are centralized here."""

    # AI backend: "claude" or "codex"
    BACKEND = "claude"

    # Model (None = use backend default)
    AI_MODEL = None

    # Codex automation policy (avoids read-only fallback in non-interactive mode due to approvals)
    CODEX_SANDBOX = "workspace-write"      # read-only | workspace-write | danger-full-access
    CODEX_APPROVAL_POLICY = "never"        # untrusted | on-failure | on-request | never
    CODEX_ENABLE_SEARCH = True             # Enable Codex native web search (--search)
    CODEX_SKIP_GIT_REPO_CHECK = True
    CODEX_BYPASS_SANDBOX = False           # Use --dangerously-bypass-approvals-and-sandbox
    CODEX_EXTRA_DIRS = []                  # Extra writable dirs passed via --add-dir

    # Claude automation policy (avoids permission popups blocking -p non-interactive mode)
    CLAUDE_SKIP_PERMISSIONS = True         # Append --dangerously-skip-permissions

    # Resolved backend executable path (set by check_backend_available)
    BACKEND_BIN = None

    # Max interaction turns per step (only Claude Code supports --max-turns)
    MAX_TURNS_PER_STEP = {
        1: 60,
        2: 30,
        3: 80,
        4: 80,
        5: 80,
        6: 80,
    }

    # Evaluation thresholds (1-10). Scores below threshold do not pass.
    THRESHOLDS = {
        1: {"coverage": 7, "quality": 7, "relevance": 7},
        2: {"novelty": 7, "feasibility": 7, "clarity": 7},
        3: {"feasibility": 7, "novelty_diff": 7, "preliminary_result": 7, "exp_plan": 7},
        4: {"correctness": 7, "completeness": 7, "documentation": 7},
        5: {"effectiveness": 7, "rigor": 7, "significance": 7, "requirement_coverage": 7},
        6: {"quality": 7, "consistency": 7, "originality": 7, "soundness": 7, "requirement_coverage": 7, "formatting": 7},
    }

    # Retry and fallback limits
    MAX_STEP_RETRIES = 5       # Max retries per step
    MAX_FALLBACK_RETRIES = 5   # Max fallbacks per step
    MAX_GLOBAL_FALLBACKS = 15  # Global fallback limit (terminate when exceeded)


# Input documents required by each step
STEP_INPUTS = {
    1: ["Input.md"],
    2: ["Input.md", "Related_work.md"],
    3: ["Input.md", "Related_work.md", "Idea.md"],
    4: ["Input.md", "Related_work.md", "Idea.md", "Methodology.md", "Initial_check.md", "Experiment.md"],
    5: ["Input.md", "Idea.md", "Methodology.md", "Experiment.md", "Codebase_guide.md"],
    6: ["Input.md", "Related_work.md", "Idea.md", "Methodology.md", "Initial_check.md", "Experiment.md", "Conclusion.md"],
}

# Expected output documents for each step
STEP_OUTPUTS = {
    1: ["Related_work.md"],
    2: ["Idea.md"],
    3: ["Methodology.md", "Initial_check.md", "Experiment.md"],
    4: ["Experiment.md", "Codebase_guide.md"],
    5: ["Experiment.md", "Conclusion.md"],
    6: ["Outline.md", "paper.tex", "paper.pdf", "references.bib"],  # Also includes figure_*.pdf, detected dynamically
}

# Fallback target: which step to return to when a step fails completely
FALLBACK_TARGETS = {
    2: 1,   # Step 2 fails -> return to Step 1 (flow: "return to 1")
    3: 2,   # Step 3 fails -> return to Step 2 (re-evaluate idea)
    4: 3,   # Step 4 fails -> return to Step 3 (flow: "return to 3")
    5: 3,   # Step 5 fails -> return to Step 3 (flow: "return to 3")
}


# ============================================================
# Logging
# ============================================================

def setup_logging(work_dir: Path) -> logging.Logger:
    log_dir = work_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = logging.getLogger("research_agent")
    logger.setLevel(logging.DEBUG)

    # File handler: log everything
    fh = logging.FileHandler(log_dir / f"research_{timestamp}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    # Console handler: log INFO and above only
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ============================================================
# AI backend invocation
# ============================================================

# Backend metadata
_BACKEND_META = {
    "claude": {
        "env_var": "CLAUDE_BIN",
        "version_arg": "--version",
        "bin_candidates_windows": ["claude.exe", "claude.cmd", "claude.bat", "claude"],
        "bin_candidates_posix": ["claude"],
        "install_hint": "npm install -g @anthropic-ai/claude-code",
    },
    "codex": {
        "env_var": "CODEX_BIN",
        "version_arg": "--version",
        "bin_candidates_windows": ["codex.exe", "codex.cmd", "codex.bat", "codex"],
        "bin_candidates_posix": ["codex"],
        "install_hint": "npm install -g @openai/codex",
    },
}


WORKFLOW_SNAPSHOT_NAME = "_workflow_flow.md"


def prepare_workflow_snapshot(work_dir: Path) -> Path:
    """Copy `flow.md` into the work directory so sandboxed backends can read it."""
    src = Path(__file__).parent.resolve() / "flow.md"
    dst = work_dir / WORKFLOW_SNAPSHOT_NAME
    shutil.copyfile(src, dst)
    return dst


class InfrastructureError(RuntimeError):
    """Raised when the backend cannot access the local workspace reliably."""


def _resolve_backend_executable(backend: str) -> str:
    """Resolve backend executable path. Prefer env var, then PATH lookup."""
    meta = _BACKEND_META[backend]

    env_var = meta.get("env_var")
    if env_var:
        env_path = (os.environ.get(env_var) or "").strip()
        if env_path:
            return env_path

    candidates = (
        meta.get("bin_candidates_windows", [])
        if os.name == "nt"
        else meta.get("bin_candidates_posix", [])
    )

    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    return ""


def _build_exec_cmd(executable: str, args: list) -> list:
    """Build a cross-platform executable command. Use `cmd /c` for .cmd/.bat on Windows."""
    if os.name == "nt" and executable.lower().endswith((".cmd", ".bat")):
        return ["cmd", "/c", executable, *args]
    return [executable, *args]


def _backend_executable(backend: str) -> str:
    """Get backend executable path. For current backend, prefer cached resolved path."""
    if backend == Config.BACKEND and Config.BACKEND_BIN:
        return Config.BACKEND_BIN

    resolved = _resolve_backend_executable(backend)
    return resolved or backend


def check_backend_available(logger: logging.Logger) -> bool:
    """Check whether the selected AI backend CLI is installed and available."""
    meta = _BACKEND_META[Config.BACKEND]

    backend_bin = _resolve_backend_executable(Config.BACKEND)
    if not backend_bin:
        logger.error(
            f"{Config.BACKEND} command not found. Please install it first:\n"
            f"  {meta['install_hint']}\n"
            f"You can also set env var {meta['env_var']} to the executable path."
        )
        return False

    version_cmd = _build_exec_cmd(backend_bin, [meta["version_arg"]])

    try:
        result = subprocess.run(
            version_cmd,
            capture_output=True, encoding="utf-8", timeout=10,
        )
    except FileNotFoundError:
        logger.error(
            f"{Config.BACKEND} command not found. Please install it first:\n"
            f"  {meta['install_hint']}\n"
            f"Currently resolved path: {backend_bin}"
        )
        return False

    if result.returncode != 0:
        logger.error(
            f"{Config.BACKEND} command exists but self-check failed (exit code={result.returncode})."
        )
        output_tail = ((result.stderr or "") or (result.stdout or ""))[-1000:]
        if output_tail:
            logger.error(f"output (tail): {output_tail}")
        return False

    Config.BACKEND_BIN = backend_bin
    logger.info(f"Using {Config.BACKEND} executable: {Config.BACKEND_BIN}")
    return True


def call_ai(prompt: str, work_dir: Path, logger: logging.Logger,
            max_turns: int = 50) -> str:
    """
    Unified AI invocation entry point. Dispatches by `Config.BACKEND`.
    Returns AI text output.
    """
    if Config.BACKEND == "claude":
        return _call_claude(prompt, work_dir, logger, max_turns)
    elif Config.BACKEND == "codex":
        return _call_codex(prompt, work_dir, logger, max_turns)
    else:
        raise ValueError(f"Unknown AI backend: {Config.BACKEND}")


def _call_claude(prompt: str, work_dir: Path, logger: logging.Logger,
                 max_turns: int) -> str:
    """
    Call Claude Code CLI with streaming JSON output.
    Uses `-p --output-format stream-json` so we can monitor progress in
    real-time while the process runs (instead of waiting for it to finish).
    Returns the final assistant text output.
    """
    claude_bin = _backend_executable("claude")
    claude_args = ["-p", "--verbose", "--output-format", "stream-json",
                   "--max-turns", str(max_turns)]
    if Config.CLAUDE_SKIP_PERMISSIONS:
        claude_args.append("--dangerously-skip-permissions")
    if Config.AI_MODEL:
        claude_args.extend(["--model", Config.AI_MODEL])
    cmd = _build_exec_cmd(claude_bin, claude_args)

    logger.debug(f"  Calling Claude Code: {' '.join(cmd)}")
    logger.debug(f"  Prompt preview: {prompt[:300]}...")

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(work_dir),
        )
    except FileNotFoundError:
        logger.error(
            "  `claude` command not found. Please ensure Claude Code CLI is installed:\n"
            "  npm install -g @anthropic-ai/claude-code\n"
            "  Or set env var CLAUDE_BIN to the executable path"
        )
        raise

    # Send prompt and close stdin so Claude starts processing
    try:
        proc.stdin.write(prompt.encode("utf-8"))
        proc.stdin.close()
    except OSError as e:
        logger.error(f"  Failed to write prompt to Claude stdin: {e}")
        proc.kill()
        raise

    # Read streaming JSON events line-by-line from stdout
    result_texts = []
    try:
        for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.debug(f"  [stream] non-JSON: {line[:200]}")
                continue
            _log_stream_event(event, logger)
            # Collect assistant text from result messages
            if event.get("type") == "result":
                result_text = event.get("result", "")
                if result_text:
                    result_texts.append(result_text)
            elif event.get("type") == "assistant" and "message" in event:
                msg = event["message"]
                if isinstance(msg, dict):
                    for block in msg.get("content", []):
                        if isinstance(block, dict) and block.get("type") == "text":
                            result_texts.append(block["text"])
    except Exception as e:
        logger.warning(f"  Error reading Claude stream: {e}")

    # Wait for process to finish
    try:
        proc.wait(timeout=14400)
    except subprocess.TimeoutExpired:
        logger.error("  Claude Code timed out (4 hours)")
        proc.kill()
        proc.wait()
        raise

    stderr_output = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    if proc.returncode != 0:
        logger.warning(f"  Claude Code exit code: {proc.returncode}")
        if stderr_output:
            logger.warning(f"  stderr: {stderr_output[:1000]}")

    output = "\n".join(result_texts)
    logger.debug(f"  Output length: {len(output)} chars")
    return output


def _log_stream_event(event: dict, logger: logging.Logger):
    """Log a Claude stream-json event in a concise, human-readable way."""
    etype = event.get("type", "")

    if etype == "system":
        # Session init info
        session_id = event.get("session_id", "")
        logger.info(f"  [stream] session started: {session_id}")

    elif etype == "assistant":
        msg = event.get("message", {})
        if isinstance(msg, dict):
            for block in msg.get("content", []):
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "tool_use":
                    tool_name = block.get("name", "?")
                    tool_input = block.get("input", {})
                    # Show a compact summary of tool input
                    summary = json.dumps(tool_input, ensure_ascii=False)
                    if len(summary) > 300:
                        summary = summary[:300] + "..."
                    logger.info(f"  [stream] tool_call: {tool_name} | {summary}")
                elif btype == "text":
                    text = block.get("text", "")
                    preview = text[:200].replace("\n", " ")
                    if len(text) > 200:
                        preview += "..."
                    logger.info(f"  [stream] assistant: {preview}")
                elif btype == "thinking":
                    text = block.get("thinking", "")
                    preview = text[:150].replace("\n", " ")
                    if len(text) > 150:
                        preview += "..."
                    logger.info(f"  [stream] thinking: {preview}")

    elif etype == "result":
        cost = event.get("cost_usd")
        duration = event.get("duration_ms")
        turns = event.get("num_turns")
        parts = []
        if turns is not None:
            parts.append(f"turns={turns}")
        if duration is not None:
            parts.append(f"duration={duration / 1000:.1f}s")
        if cost is not None:
            parts.append(f"cost=${cost:.4f}")
        logger.info(f"  [stream] result: {', '.join(parts)}")

    # Silently skip other event types (e.g. tool_result content)


def _call_codex(prompt: str, work_dir: Path, logger: logging.Logger,
                max_turns: int) -> str:
    """
    Call OpenAI Codex CLI with real-time stderr streaming.
    Uses Popen so we can log Codex's stderr progress lines as they arrive
    instead of buffering everything until the process exits.
    Codex prints progress/session info to stderr and final output to stdout.
    """
    codex_bin = _backend_executable("codex")
    codex_args = []
    if Config.CODEX_BYPASS_SANDBOX:
        codex_args.append("--dangerously-bypass-approvals-and-sandbox")
    else:
        # Note: --ask-for-approval is a top-level Codex argument and must appear before `exec`
        codex_args.extend([
            "--ask-for-approval", Config.CODEX_APPROVAL_POLICY,
            "--sandbox", Config.CODEX_SANDBOX,
        ])
    if Config.CODEX_ENABLE_SEARCH:
        codex_args.append("--search")
    codex_args.extend([
        "exec",
        "--cd", str(work_dir),
    ])
    for extra_dir in Config.CODEX_EXTRA_DIRS:
        codex_args.extend(["--add-dir", str(extra_dir)])
    if Config.CODEX_SKIP_GIT_REPO_CHECK:
        codex_args.append("--skip-git-repo-check")
    if Config.AI_MODEL:
        codex_args.extend(["--model", Config.AI_MODEL])
    codex_args.append(prompt)
    cmd = _build_exec_cmd(codex_bin, codex_args)

    if Config.CODEX_BYPASS_SANDBOX:
        logger.debug("  Calling Codex: codex --dangerously-bypass-approvals-and-sandbox exec ...")
    else:
        logger.debug(
            "  Calling Codex: codex --ask-for-approval %s --sandbox %s exec ...",
            Config.CODEX_APPROVAL_POLICY,
            Config.CODEX_SANDBOX,
        )
    logger.debug(f"  Prompt preview: {prompt[:300]}...")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(work_dir),
        )
    except FileNotFoundError:
        logger.error(
            "  `codex` command not found. Please ensure OpenAI Codex CLI is installed:\n"
            "  npm install -g @openai/codex\n"
            "  Or set env var CODEX_BIN to the executable path"
        )
        raise

    # Stream stderr in a background thread so it doesn't block stdout reading
    import threading

    stderr_lines = []

    def _drain_stderr():
        for raw_line in proc.stderr:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            if line:
                stderr_lines.append(line)
                logger.info(f"  [codex] {line}")

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    # Read stdout (final output) line-by-line
    stdout_chunks = []
    try:
        for raw_line in proc.stdout:
            stdout_chunks.append(raw_line.decode("utf-8", errors="replace"))
    except Exception as e:
        logger.warning(f"  Error reading Codex stdout: {e}")

    # Wait for process to finish
    try:
        proc.wait(timeout=14400)
    except subprocess.TimeoutExpired:
        logger.error("  Codex timed out (4 hours)")
        proc.kill()
        proc.wait()
        raise

    stderr_thread.join(timeout=5)

    _debug_log_codex_runtime_info("\n".join(stderr_lines), logger)

    if proc.returncode != 0:
        logger.warning(f"  Codex exit code: {proc.returncode}")
        if stderr_lines:
            tail = "\n".join(stderr_lines[-20:])
            logger.warning(f"  stderr (tail): {tail}")

    output = "".join(stdout_chunks)
    logger.debug(f"  Output length: {len(output)} chars")
    return output


def _output_suggests_workspace_access_issue(output: str) -> bool:
    text = (output or "").lower()
    patterns = [
        "landlock",
        "sandbox restriction",
        "cannot read workspace files",
        "could not read",
        "local file access is blocked",
        "environment prevented all local file reads and writes",
        "unable to open",
    ]
    return any(pattern in text for pattern in patterns)


def _debug_log_codex_runtime_info(stderr_text: str, logger: logging.Logger) -> None:
    """Extract and print key session info from Codex stderr."""
    if not stderr_text:
        return

    fields = [
        "model",
        "provider",
        "approval",
        "sandbox",
        "reasoning effort",
        "reasoning summaries",
        "session id",
    ]

    parsed = {}
    for line in stderr_text.splitlines():
        text = line.strip()
        if not text or ":" not in text:
            continue

        m = re.match(r"^([A-Za-z ]+):\s*(.+)$", text)
        if not m:
            continue

        key = m.group(1).strip().lower()
        value = m.group(2).strip()
        if key in fields:
            parsed[key] = value

    if not parsed:
        return

    logger.debug("  Codex runtime info:")
    for key in fields:
        if key in parsed:
            logger.debug("    %s: %s", key, parsed[key])


# ============================================================
# Rubrics
# ============================================================

RUBRICS = {
    1: """Evaluate literature survey quality:
- coverage: Does it cover major research directions and important recent papers in the field?
- quality: Is the paper analysis in-depth? Are key contributions and limitations identified?
- relevance: Are selected papers highly relevant to the research topic?
Threshold: proceed only if all dimensions are >= {min_score}; otherwise retry.""",

    2: """Evaluate the research idea:
- novelty: Compared with existing methods, does the idea contain substantial innovation?
- feasibility: Is the idea implementable under current technical conditions?
- clarity: Is the problem definition clear? Are "what to solve, why important, and existing limitations" all clear?
Threshold: proceed only if all dimensions are >= {min_score}.
If novelty is severely insufficient (< 5), fallback to Step 1 for re-survey is recommended.""",

    3: """Evaluate solution method, preliminary validation, and experimental plan:
- feasibility: Is method complexity manageable? Is implementation feasible?
- novelty_diff: Is the method sufficiently different from methods in the literature?
- preliminary_result: Do small experiments preliminarily validate component functionality and method effectiveness?
- exp_plan: Is the experimental plan in Experiment.md well-structured with clear metrics, baselines, and ablation design?
Threshold: proceed only if all dimensions are >= {min_score}.
If difference from existing methods is too small (novelty_diff < 5), fallback is recommended.""",

    4: """Evaluate baseline implementation and code documentation:
- correctness: Does the baseline implementation run end-to-end without errors? Are module-level checks passing?
- completeness: Are all sub-methods implemented in their basic versions? Are baseline results recorded?
- documentation: Does Codebase_guide.md clearly document all critical code files and code blocks with sufficient detail?
Threshold: proceed only if all dimensions are >= {min_score}.
If implementation has fundamental errors, fallback to Step 3 to revise the method is recommended.""",

    5: """Evaluate full experimental results and ablation studies:
- effectiveness: Does the method outperform baselines on primary metrics?
- rigor: Are baseline comparisons and ablation studies sufficient?
- significance: Are improvements practically meaningful (not trivial gains)?
- requirement_coverage: Do the experiments cover all core experimental objectives specified in Input.md? If any objective is missing, is there a documented justification?
Threshold: proceed only if all dimensions are >= {min_score}.
If the method does not outperform any baseline, fallback to Step 3 to improve the method is recommended.""",

    6: """Evaluate paper quality:
- quality: Is writing clear and logically rigorous?
- consistency: Are terminology, symbols, and argumentation consistent throughout?
- originality: Do Abstract and Introduction highlight novelty?
- soundness: Is the paper scientifically sound end-to-end? Are claims in the Introduction well-supported by the methodology and experiments? Are experimental conclusions justified by the reported results? Are there any logical gaps, unsupported assertions, or mismatches between stated contributions and actual evidence?
- requirement_coverage: Does the paper address all research objectives from Input.md? Are unmet objectives acknowledged in Limitations?
- formatting: Does it comply with the paper template requirements?
Threshold: proceed only if all dimensions are >= {min_score}.""",
}


# ============================================================
# Step execution
# ============================================================

def build_executor_prompt(step_num: int, retry_context: str = "") -> str:
    """Build the executor prompt."""
    flow_file = WORKFLOW_SNAPSHOT_NAME
    input_files = ", ".join(STEP_INPUTS.get(step_num, []))
    output_files = ", ".join(STEP_OUTPUTS.get(step_num, []))

    prompt = (
        # f"You are a professional research AI assistant executing a systematic research workflow.\n\n"
        f"Please first read {flow_file} to understand the complete workflow, "
        f"then strictly execute [Step {step_num}] in it.\n\n"
        f"Input documents to reference: {input_files}\n"
        f"Output documents to produce: {output_files}\n\n"
        f"Important rules:\n"
        f"1. Read the input documents above first to obtain context;\n"
        f"2. Write all output documents directly to the current working directory;\n"
        f"3. If paper or information search is needed, use web search;\n"
        f"4. If programming experiments are needed, directly write and execute code;\n"
        f"5. Complete required iterative refinements within the step autonomously;\n"
        f"6. Ensure outputs strictly follow requirements in {flow_file};\n"
        f"7. Do not ask the user for any input; you are a fully automated agent;\n"
        f"8. If information is missing, fill gaps based on your research expertise.\n"
    )

    if retry_context:
        prompt += (
            f"\n{'='*40}\n"
            f"Note: This is a retry/fallback. Improve your output based on the feedback below:\n"
            f"{retry_context}\n"
            f"{'='*40}\n"
        )

    return prompt


def execute_step(step_num: int, work_dir: Path,
                 logger: logging.Logger, retry_context: str = "") -> str:
    """
    Invoke the AI backend to execute one step in the research workflow.
    Returns AI text output.
    """
    prompt = build_executor_prompt(step_num, retry_context)
    max_turns = Config.MAX_TURNS_PER_STEP.get(step_num, 50)

    logger.info(f"  Executing step {step_num} via {Config.BACKEND} (max_turns={max_turns})...")
    result = call_ai(prompt, work_dir, logger, max_turns)
    missing_outputs = [
        fname for fname in STEP_OUTPUTS.get(step_num, [])
        if not (work_dir / fname).exists()
    ]
    if missing_outputs:
        missing_text = ", ".join(missing_outputs)
        if _output_suggests_workspace_access_issue(result):
            raise InfrastructureError(
                f"Step {step_num} could not access the workspace and did not produce: {missing_text}. "
                f"This usually means the Codex sandbox cannot operate on the current filesystem. "
                f"Try `--codex-bypass-sandbox`, `--codex-sandbox danger-full-access`, or move the "
                f"workspace to a local disk before rerunning."
            )
        raise RuntimeError(
            f"Step {step_num} completed without writing expected outputs: {missing_text}"
        )
    logger.info(f"  Execution completed.")
    return result


# ============================================================
# Evaluator
# ============================================================

def evaluate_step(step_num: int, work_dir: Path,
                  logger: logging.Logger) -> dict:
    """
    Invoke the AI backend to evaluate step outputs.
    AI writes evaluation to `_eval_step_N.json`, then this function reads/parses it.
    Returns: {"scores": {...}, "decision": "...", "reason": "...", "suggestions": "..."}
    """
    thresholds = Config.THRESHOLDS.get(step_num, {})
    min_score = min(thresholds.values()) if thresholds else 6
    rubric = RUBRICS.get(step_num, "").format(min_score=min_score)

    input_files = ", ".join(STEP_INPUTS.get(step_num, []))
    output_files = ", ".join(STEP_OUTPUTS.get(step_num, []))
    eval_file = f"_eval_step_{step_num}.json"
    eval_path = work_dir / eval_file

    # Clean previous evaluation file
    if eval_path.exists():
        eval_path.unlink()

    prompt = (
        f"You are an independent expert evaluator of research quality.\n"
        f"Your task is to objectively evaluate the output quality of an AI research assistant at [Step {step_num}].\n"
        f"Score strictly by the rubric (1-10). Do not be overly lenient.\n\n"
        f"Input documents (for reference): {input_files}\n"
        f"Output documents (to evaluate): {output_files}\n\n"
        f"Please read all documents above first, then evaluate according to the rubric below:\n\n"
        f"{rubric}\n\n"
        f"After evaluation, write results to file {eval_file}, strictly using this JSON format:\n"
        f'{{\n'
        f'    "scores": {{"metric_name_in_english": score, ...}},\n'
        f'    "decision": "proceed or retry or fallback or abort",\n'
        f'    "reason": "decision rationale (1-2 sentences)",\n'
        f'    "suggestions": "if not proceed, provide specific improvement suggestions"\n'
        f'}}\n\n'
        f"Use English metric names in scores (as listed in the rubric).\n"
        f"Decision meanings: proceed=pass to next step, retry=retry current step, "
        f"fallback=return to an upstream step, abort=terminate the whole research direction.\n"
    )

    logger.info(f"  Evaluating Step {step_num} via {Config.BACKEND}...")
    call_ai(prompt, work_dir, logger, max_turns=20)

    # Read evaluation result
    if eval_path.exists():
        try:
            evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
            logger.info(
                f"  Evaluation: scores={evaluation.get('scores')}, "
                f"decision={evaluation.get('decision')}, "
                f"reason={evaluation.get('reason')}"
            )
            # Clean temporary evaluation file
            eval_path.unlink(missing_ok=True)
            return evaluation
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"  Failed to parse evaluation file: {e}")

    logger.warning("  Evaluation file missing or unparsable; defaulting to retry")
    return {"scores": {}, "decision": "retry", "reason": "Evaluation result was not generated correctly"}


def check_thresholds(step_num: int, evaluation: dict) -> bool:
    """Check whether all scores meet thresholds. Returns True if passed."""
    scores = evaluation.get("scores", {})
    thresholds = Config.THRESHOLDS.get(step_num, {})
    for metric, min_val in thresholds.items():
        actual = scores.get(metric, 0)
        if actual < min_val:
            return False
    return True


# ============================================================
# Main orchestrator
# ============================================================

def run_research(work_dir: Path, start_step: int = 1, end_step: int = 6):
    """Main flow: orchestrate execution, evaluation, retries, and fallbacks up to end_step."""

    logger = setup_logging(work_dir)
    logger.info("=" * 60)
    logger.info(f"Automated Research Agent started (backend: {Config.BACKEND})")
    logger.info(f"Working directory: {work_dir.resolve()}")
    logger.info(f"Start step: {start_step}")
    logger.info(f"End step: {end_step}")
    if Config.AI_MODEL:
        logger.info(f"Specified model: {Config.AI_MODEL}")
    logger.info("=" * 60)

    # Check required files
    # flow.md is in project root
    flow_path = Path(__file__).parent.resolve() / "flow.md"
    # Input.md is under work_dir
    input_path = work_dir / "Input.md"
    if not flow_path.exists():
        logger.error(f"flow.md not found. Please confirm project root: {flow_path.parent}")
        return
    if not input_path.exists():
        logger.error("Input.md not found. Please create it first (original idea, references, experiment thoughts, paper format).")
        return

    # Check whether AI backend CLI is available
    if not check_backend_available(logger):
        return

    try:
        snapshot_path = prepare_workflow_snapshot(work_dir)
        logger.info(f"Workflow snapshot prepared: {snapshot_path.name}")
    except Exception as e:
        logger.error(f"Failed to prepare workflow snapshot in work dir: {e}")
        return

    # State tracking
    step = start_step
    retry_counts = {i: 0 for i in range(1, 7)}
    fallback_counts = {i: 0 for i in range(1, 7)}
    global_fallbacks = 0
    retry_context = ""

    while step <= end_step:
        logger.info("")
        logger.info("=" * 60)
        logger.info(
            f">>> Step {step}  "
            f"[retry {retry_counts[step]}/{Config.MAX_STEP_RETRIES}, "
            f"fallback {fallback_counts[step]}/{Config.MAX_FALLBACK_RETRIES}, "
            f"global fallback {global_fallbacks}/{Config.MAX_GLOBAL_FALLBACKS}]"
        )
        logger.info("=" * 60)

        # ---------- Execute ----------
        try:
            summary = execute_step(step, work_dir, logger, retry_context)
            logger.info(f"  Summary: {summary[:300]}...")
        except InfrastructureError as e:
            logger.error(f"  Infrastructure error executing step {step}: {e}")
            logger.error("  Terminating workflow early because retrying will not fix sandbox/file-access issues.")
            break
        except Exception as e:
            logger.error(f"  Error executing step {step}: {e}")
            retry_context = f"Execution error: {e}"
            retry_counts[step] += 1
            if retry_counts[step] < Config.MAX_STEP_RETRIES:
                logger.info(f"  Will retry step {step}...")
                continue
            else:
                # Try fallback
                target = FALLBACK_TARGETS.get(step)
                if target and global_fallbacks < Config.MAX_GLOBAL_FALLBACKS:
                    logger.info(f"  Retries exhausted, falling back to step {target}")
                    fallback_counts[step] += 1
                    global_fallbacks += 1
                    origin_step = step
                    step = target
                    retry_counts[step] = 0
                    retry_context = f"Fallback from downstream of step {origin_step}; please improve direction."
                    continue
                else:
                    logger.error("  Cannot continue; terminating workflow.")
                    break

        # ---------- Evaluate ----------
        logger.info(f"  Evaluating Step {step}...")
        try:
            evaluation = evaluate_step(step, work_dir, logger)
        except Exception as e:
            logger.error(f"  Evaluation error: {e}; defaulting to retry")
            evaluation = {
                "scores": {}, "decision": "retry",
                "reason": f"Evaluation error ({e}); default retry",
            }

        decision = evaluation.get("decision", "retry")
        reason = evaluation.get("reason", "")
        suggestions = evaluation.get("suggestions", "")
        scores = evaluation.get("scores", {})

        # If evaluator says proceed but thresholds are unmet, downgrade to retry
        if decision == "proceed" and not check_thresholds(step, evaluation):
            logger.warning(
                f"  Evaluator suggested proceed but thresholds unmet (scores={scores}); downgraded to retry"
            )
            decision = "retry"
            reason += " (thresholds unmet, auto-downgraded)"

        logger.info(f"  Decision: {decision}")
        logger.info(f"  Reason:   {reason}")
        if suggestions:
            logger.info(f"  Suggest:  {suggestions}")

        # ---------- Decide ----------
        if decision == "proceed":
            logger.info(f"  Step {step} passed! Moving to next step.")
            retry_context = ""
            retry_counts[step] = 0
            step += 1

        elif decision == "retry":
            retry_counts[step] += 1
            if retry_counts[step] < Config.MAX_STEP_RETRIES:
                retry_context = f"Evaluation feedback: {reason}\nImprovement suggestions: {suggestions}"
                logger.info(
                    f"  Retrying step {step} "
                    f"({retry_counts[step]}/{Config.MAX_STEP_RETRIES})"
                )
            else:
                # Retries exhausted; try fallback
                target = FALLBACK_TARGETS.get(step)
                if target and fallback_counts[step] < Config.MAX_FALLBACK_RETRIES \
                        and global_fallbacks < Config.MAX_GLOBAL_FALLBACKS:
                    logger.info(
                        f"  Step {step} retries exhausted, fallback to step {target}"
                    )
                    fallback_counts[step] += 1
                    global_fallbacks += 1
                    retry_counts[step] = 0
                    retry_counts[target] = 0
                    origin_step = step
                    step = target
                    retry_context = (
                        f"Fallback from downstream of Step {origin_step}.\n"
                        f"Reason: {reason}\nImprovement suggestions: {suggestions}"
                    )
                else:
                    logger.error("  Both retries and fallbacks are exhausted; terminating workflow.")
                    break

        elif decision == "fallback":
            target = FALLBACK_TARGETS.get(step)
            if target and fallback_counts[step] < Config.MAX_FALLBACK_RETRIES \
                    and global_fallbacks < Config.MAX_GLOBAL_FALLBACKS:
                logger.info(f"  Evaluator requested fallback: step {step} -> step {target}")
                fallback_counts[step] += 1
                global_fallbacks += 1
                retry_counts[step] = 0
                retry_counts[target] = 0
                origin_step = step
                step = target
                retry_context = (
                    f"Fallback from downstream of Step {origin_step}.\n"
                    f"Reason: {reason}\nImprovement suggestions: {suggestions}"
                )
            else:
                logger.error("  Cannot fallback; terminating workflow.")
                break

        elif decision == "abort":
            logger.error(f"  Evaluator suggested abort: {reason}")
            logger.error("  Terminating workflow.")
            break

        # Global safety check
        if global_fallbacks >= Config.MAX_GLOBAL_FALLBACKS:
            logger.error(
                f"  Global fallback count reached limit ({Config.MAX_GLOBAL_FALLBACKS}); terminating workflow."
            )
            break

    # ---------- Finish ----------
    logger.info("")
    logger.info("=" * 60)
    if step > end_step:
        logger.info("Research workflow completed successfully!")
    else:
        logger.info(f"Research workflow terminated at step {step}.")
    logger.info("=" * 60)

    # List all output files
    logger.info("Output documents:")
    all_outputs = set()
    for s in range(1, end_step + 1):
        all_outputs.update(STEP_OUTPUTS.get(s, []))
    # Include dynamic files (figure_*.pdf)
    for f in work_dir.glob("figure_*.pdf"):
        all_outputs.add(f.name)

    for fname in sorted(all_outputs):
        fpath = work_dir / fname
        icon = "[OK]" if fpath.exists() else "[--]"
        logger.info(f"  {icon} {fname}")


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Automated Research Agent - supports Claude Code / OpenAI Codex dual backends"
    )
    parser.add_argument(
        "--backend", type=str, default="claude",
        choices=["claude", "codex"],
        help="AI backend: claude (Claude Code) or codex (OpenAI Codex) (default: codex)",
    )
    parser.add_argument(
        "--work-dir", type=str, default="./work3",
        help="Working directory containing Input.md and intermediate outputs (default: ./work; flow.md is in project root)",
    )
    parser.add_argument(
        "--start-step", type=int, default=1, choices=[1, 2, 3, 4, 5, 6],
        help="Step number to start from (default: 1, for resuming runs)",
    )
    parser.add_argument(
        "--end-step", type=int, default=6, choices=[1, 2, 3, 4, 5, 6],
        help="Step number to end at (default: 6)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Specify AI model (default: backend configuration)",
    )
    parser.add_argument(
        "--claude-skip-permissions",
        action=argparse.BooleanOptionalAction,
        default=Config.CLAUDE_SKIP_PERMISSIONS,
        help=(
            "Whether Claude should skip permission confirmation prompts. "
            "When enabled, --dangerously-skip-permissions is appended, "
            "which is suitable for fully automated non-interactive runs like run_research."
        ),
    )
    parser.add_argument(
        "--codex-sandbox", type=str, default=Config.CODEX_SANDBOX,
        choices=["read-only", "workspace-write", "danger-full-access"],
        help="Codex sandbox mode (default: workspace-write)",
    )
    parser.add_argument(
        "--codex-approval-policy", type=str, default=Config.CODEX_APPROVAL_POLICY,
        choices=["untrusted", "on-failure", "on-request", "never"],
        help="Codex approval policy (default: never)",
    )
    parser.add_argument(
        "--codex-search",
        action=argparse.BooleanOptionalAction,
        default=Config.CODEX_ENABLE_SEARCH,
        help="Whether to enable Codex native web search (default: enabled)",
    )
    parser.add_argument(
        "--codex-bypass-sandbox",
        action=argparse.BooleanOptionalAction,
        default=Config.CODEX_BYPASS_SANDBOX,
        help=(
            "Run Codex with --dangerously-bypass-approvals-and-sandbox. "
            "Use this only in an already isolated environment, such as a trusted container or VM."
        ),
    )
    parser.add_argument(
        "--codex-add-dir", action="append", default=[],
        help="Extra directory to expose to Codex via --add-dir (repeatable)",
    )
    args = parser.parse_args()
    if args.end_step < args.start_step:
        parser.error("--end-step must be greater than or equal to --start-step")

    Config.BACKEND = args.backend
    if args.model:
        Config.AI_MODEL = args.model
    Config.CLAUDE_SKIP_PERMISSIONS = args.claude_skip_permissions
    Config.CODEX_SANDBOX = args.codex_sandbox
    Config.CODEX_APPROVAL_POLICY = args.codex_approval_policy
    Config.CODEX_ENABLE_SEARCH = args.codex_search
    Config.CODEX_BYPASS_SANDBOX = args.codex_bypass_sandbox
    Config.CODEX_EXTRA_DIRS = [Path(p).resolve() for p in args.codex_add_dir]

    work_dir = Path(args.work_dir).resolve()
    run_research(work_dir, start_step=args.start_step, end_step=args.end_step)


if __name__ == "__main__":
    main()
