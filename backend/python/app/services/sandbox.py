import re
import subprocess
import sys
import time

from app.config import settings
from app.models.execution_result import ExecutionResult
from app.services.sandbox_preamble import SANDBOX_PREAMBLE

PLOT_PATTERN = re.compile(r"__KATA_PLOT_(\d+)__:(.*?):__END_KATA_PLOT__", re.DOTALL)
METRIC_PATTERN = re.compile(r"__KATA_METRIC__:(.*?):(.*?):__END_KATA_METRIC__")
TENSOR_PATTERN = re.compile(r"__KATA_TENSOR__:(.*?):__END_KATA_TENSOR__", re.DOTALL)


def _extract_structured(stdout: str) -> tuple[str, list[dict], dict, list[dict]]:
    """Extract plots, metrics, and tensors from stdout sentinel markers."""
    plots: list[dict] = []
    for match in PLOT_PATTERN.finditer(stdout):
        plots.append({
            "index": int(match.group(1)),
            "data": f"data:image/png;base64,{match.group(2)}",
            "format": "png",
        })
    stdout = PLOT_PATTERN.sub("", stdout)

    metrics: dict = {}
    for match in METRIC_PATTERN.finditer(stdout):
        key = match.group(1)
        raw = match.group(2)
        try:
            metrics[key] = float(raw)
        except ValueError:
            metrics[key] = raw
    stdout = METRIC_PATTERN.sub("", stdout)

    import json
    tensors: list[dict] = []
    for match in TENSOR_PATTERN.finditer(stdout):
        try:
            tensors.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            pass
    stdout = TENSOR_PATTERN.sub("", stdout)

    # Clean up extra blank lines left by sentinel removal
    stdout = re.sub(r"\n{3,}", "\n\n", stdout).strip()
    return stdout, plots, metrics, tensors


async def execute_code(code: str, kata_id: str) -> ExecutionResult:
    full_code = SANDBOX_PREAMBLE + "\n" + code
    start = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True,
            text=True,
            timeout=settings.sandbox_timeout_seconds,
            shell=False,
        )
        elapsed = (time.perf_counter() - start) * 1000
        stdout, plots, metrics, tensors = _extract_structured(result.stdout)
        return ExecutionResult(
            stdout=stdout,
            stderr=result.stderr,
            error=result.stderr if result.returncode != 0 else None,
            execution_time_ms=round(elapsed, 2),
            plots=plots,
            metrics=metrics,
            tensors=tensors,
        )
    except subprocess.TimeoutExpired:
        elapsed = (time.perf_counter() - start) * 1000
        return ExecutionResult(
            error=f"Execution timed out after {settings.sandbox_timeout_seconds} seconds. "
            "Your code took too long to run. Try reducing the number of iterations or data size.",
            execution_time_ms=round(elapsed, 2),
        )
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return ExecutionResult(
            error=f"Execution failed: {str(e)}",
            execution_time_ms=round(elapsed, 2),
        )
