// In-browser Python execution via Pyodide. Mirrors the SSECallbacks shape
// from sse-client.ts so the workspace component can swap implementations
// without changing its callback wiring.

import type { SSECallbacks } from "./sse-client";
import type { ExecutionResult, PlotData, TensorData } from "./types";

const PLOT_RE = /__KATA_PLOT_(\d+)__:(.*):__END_KATA_PLOT__/;
const METRIC_RE = /__KATA_METRIC__:(.*?):(.*?):__END_KATA_METRIC__/;
const TENSOR_RE = /__KATA_TENSOR__:(.*):__END_KATA_TENSOR__/;

const PYODIDE_VERSION = "0.26.4";
const PYODIDE_INDEX = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

// Mirrors SANDBOX_PREAMBLE from backend/python/app/services/sandbox_preamble.py
// Same sentinel format so the regex parsing in kata-workspace.tsx is unchanged.
const SANDBOX_PREAMBLE = `
import matplotlib as _mpl
_mpl.use('Agg')
import matplotlib.pyplot as _plt_module
import io as _io
import base64 as _b64

_plot_counter = [0]
_orig_show = _plt_module.show

def _capture_show(*args, **kwargs):
    import matplotlib.pyplot as plt
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        buf = _io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        encoded = _b64.b64encode(buf.read()).decode('ascii')
        print(f'__KATA_PLOT_{_plot_counter[0]}__:{encoded}:__END_KATA_PLOT__')
        _plot_counter[0] += 1
        buf.close()
    plt.close('all')

_plt_module.show = _capture_show


def kata_metric(name, value):
    print(f'__KATA_METRIC__:{name}:{value}:__END_KATA_METRIC__')


def kata_tensor(name, arr, max_rows=20, max_cols=20):
    import numpy as _np
    import json as _json
    if hasattr(arr, 'detach'):
        arr = arr.detach().cpu().numpy()
    arr = _np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])[:max_rows]
    arr = arr[:max_rows, :max_cols]
    data = {
        "name": name,
        "shape": list(arr.shape),
        "values": arr.tolist(),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    print(f'__KATA_TENSOR__:{_json.dumps(data)}:__END_KATA_TENSOR__')
`;

interface PyodideInstance {
  loadPackagesFromImports: (code: string) => Promise<void>;
  loadPackage: (pkg: string | string[]) => Promise<void>;
  runPythonAsync: (code: string) => Promise<unknown>;
  setStdout: (opts: { batched?: (s: string) => void }) => void;
  setStderr: (opts: { batched?: (s: string) => void }) => void;
}

declare global {
  interface Window {
    loadPyodide?: (cfg: { indexURL: string }) => Promise<PyodideInstance>;
  }
}

let pyodidePromise: Promise<PyodideInstance> | null = null;

function injectPyodideScript(): Promise<void> {
  return new Promise((resolve, reject) => {
    if (window.loadPyodide) return resolve();
    const script = document.createElement("script");
    script.src = `${PYODIDE_INDEX}pyodide.js`;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("Failed to load Pyodide script"));
    document.head.appendChild(script);
  });
}

async function getPyodide(): Promise<PyodideInstance> {
  if (!pyodidePromise) {
    pyodidePromise = (async () => {
      await injectPyodideScript();
      if (!window.loadPyodide) {
        throw new Error("Pyodide failed to attach to window");
      }
      const py = await window.loadPyodide({ indexURL: PYODIDE_INDEX });
      // Most katas need at least these — preload to avoid mid-run delays.
      await py.loadPackage(["numpy", "matplotlib"]);
      return py;
    })();
  }
  return pyodidePromise;
}

// Sentinel-aware line dispatcher — same logic as execute_stream.py:42-49
function dispatchLine(line: string, callbacks: SSECallbacks): void {
  if (line.includes("__KATA_PLOT_")) callbacks.onPlot(line);
  else if (line.includes("__KATA_METRIC__")) callbacks.onMetric(line);
  else if (line.includes("__KATA_TENSOR__")) callbacks.onTensor(line);
  else callbacks.onStdout(line);
}

export async function runInPyodide(
  code: string,
  callbacks: SSECallbacks,
): Promise<void> {
  const start = performance.now();
  let py: PyodideInstance;
  try {
    py = await getPyodide();
  } catch (e) {
    callbacks.onError(e instanceof Error ? e.message : String(e));
    callbacks.onDone(performance.now() - start);
    return;
  }

  py.setStdout({ batched: (s) => dispatchLine(s, callbacks) });
  py.setStderr({ batched: (s) => callbacks.onStderr(s) });

  const fullCode = SANDBOX_PREAMBLE + "\n" + code;

  try {
    // loadPackagesFromImports auto-loads numpy, pandas, scikit-learn,
    // matplotlib, etc. on demand. PyTorch is supported but not auto-loaded
    // by this helper for all import styles, so we explicitly nudge it.
    if (/\bimport\s+torch\b|\bfrom\s+torch\b/.test(code)) {
      try {
        await py.loadPackage("torch");
      } catch {
        callbacks.onStderr(
          "Note: torch is not yet supported in Pyodide for this build — clone the repo to run torch katas locally.",
        );
      }
    }
    await py.loadPackagesFromImports(fullCode);
    await py.runPythonAsync(fullCode);
  } catch (e) {
    callbacks.onError(e instanceof Error ? e.message : String(e));
  } finally {
    callbacks.onDone(performance.now() - start);
  }
}

// One-shot bridge: runs code in Pyodide and aggregates streamed events
// into a single ExecutionResult, matching the response shape that
// apiPost("/execute") returns from the FastAPI backend in dev.
export async function runInPyodideOneShot(code: string): Promise<ExecutionResult> {
  const result: ExecutionResult = {
    stdout: "",
    stderr: "",
    error: null,
    execution_time_ms: 0,
    metrics: {},
    plots: [],
    tensors: [],
  };

  await new Promise<void>((resolve) => {
    runInPyodide(code, {
      onStdout: (line) => {
        result.stdout += line + "\n";
      },
      onStderr: (line) => {
        result.stderr += line + "\n";
      },
      onPlot: (raw) => {
        const m = raw.match(PLOT_RE);
        if (!m) return;
        const plot: PlotData = {
          index: parseInt(m[1]),
          data: `data:image/png;base64,${m[2]}`,
          format: "png",
        };
        result.plots.push(plot);
      },
      onMetric: (raw) => {
        const m = raw.match(METRIC_RE);
        if (!m) return;
        const val = parseFloat(m[2]);
        result.metrics[m[1]] = isNaN(val) ? m[2] : val;
      },
      onTensor: (raw) => {
        const m = raw.match(TENSOR_RE);
        if (!m) return;
        try {
          const tensor: TensorData = JSON.parse(m[1]);
          result.tensors.push(tensor);
        } catch {
          /* skip malformed */
        }
      },
      onError: (msg) => {
        result.error = msg;
      },
      onDone: (ms) => {
        result.execution_time_ms = ms;
        resolve();
      },
    });
  });

  return result;
}
