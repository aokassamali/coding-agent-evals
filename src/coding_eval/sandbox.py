from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int


def run_pytest(code: str, tests: str, timeout_s: int = 10) -> ExecResult:
    """
    Run pytest on the provided solution code + tests in a temp directory.

    Important behavior:
    - Timeouts are returned as ExecResult(exit_code=124) rather than raising,
      so the caller can classify as failure_mode="timeout" and keep going.
    """
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        test_file = td_path / "test_solution.py"
        test_file.write_text(code + "\n\n" + tests + "\n", encoding="utf-8")

        try:
            p = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=str(td_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s,
            )
            return ExecResult(stdout=p.stdout or "", stderr=p.stderr or "", exit_code=int(p.returncode))

        except subprocess.TimeoutExpired as e:
            # Normalize timeout into a failure result instead of crashing the harness.
            out = (e.stdout or "") if isinstance(e.stdout, str) else ""
            err = (e.stderr or "") if isinstance(e.stderr, str) else ""
            # Add a marker so classify_failure can detect it reliably.
            err = (err + "\n\nPYTEST_TIMEOUT").strip()
            return ExecResult(stdout=out, stderr=err, exit_code=124)

