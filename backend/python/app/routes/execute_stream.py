import asyncio
import os
import subprocess
import sys
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.config import settings
from app.models.execution_result import ExecutionRequest
from app.services.sandbox_preamble import SANDBOX_PREAMBLE

router = APIRouter()


@router.post("/execute/stream")
async def stream_code(request: ExecutionRequest):
    full_code = SANDBOX_PREAMBLE + "\n" + request.code

    async def event_generator():
        start = time.perf_counter()
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", full_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        try:
            while True:
                line = proc.stdout.readline()
                if line == "" and proc.poll() is not None:
                    break
                if not line:
                    await asyncio.sleep(0.01)
                    continue

                stripped = line.rstrip("\n")
                if "__KATA_PLOT_" in stripped:
                    yield f"event: plot\ndata: {stripped}\n\n"
                elif "__KATA_METRIC__" in stripped:
                    yield f"event: metric\ndata: {stripped}\n\n"
                elif "__KATA_TENSOR__" in stripped:
                    yield f"event: tensor\ndata: {stripped}\n\n"
                else:
                    yield f"event: stdout\ndata: {stripped}\n\n"

                await asyncio.sleep(0)

            stderr = proc.stderr.read()
            elapsed = (time.perf_counter() - start) * 1000

            if stderr:
                for line in stderr.strip().split("\n"):
                    yield f"event: stderr\ndata: {line}\n\n"

            if proc.returncode != 0 and stderr:
                yield f"event: error\ndata: {stderr.strip()}\n\n"

            yield f"event: done\ndata: {elapsed:.2f}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"
            elapsed = (time.perf_counter() - start) * 1000
            yield f"event: done\ndata: {elapsed:.2f}\n\n"
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
