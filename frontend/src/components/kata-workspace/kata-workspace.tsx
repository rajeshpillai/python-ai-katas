import { createSignal, createEffect, on, Show } from "solid-js";
import Resizable from "@corvu/resizable";
import CodePanel from "./code-panel";
import OutputPanel from "./output-panel";
import { apiPost } from "../../lib/api-client";
import { executeStream } from "../../lib/sse-client";
import type { ExecutionResult, PlotData, TensorData } from "../../lib/types";
import "./kata-workspace.css";

type MaximizedPanel = "code" | "output" | null;

const PLOT_RE = /__KATA_PLOT_(\d+)__:(.*):__END_KATA_PLOT__/;
const METRIC_RE = /__KATA_METRIC__:(.*?):(.*?):__END_KATA_METRIC__/;
const TENSOR_RE = /__KATA_TENSOR__:(.*):__END_KATA_TENSOR__/;

const outputCache = new Map<string, ExecutionResult | null>();

interface KataWorkspaceProps {
  kataId?: string;
  cacheKey?: string;
  defaultCode?: string;
}

export default function KataWorkspace(props: KataWorkspaceProps) {
  const [maximized, setMaximized] = createSignal<MaximizedPanel>(null);
  const [output, setOutput] = createSignal<ExecutionResult | null>(null);
  const [running, setRunning] = createSignal(false);
  const [streaming, setStreaming] = createSignal(false);

  // Save/restore output when navigating between katas
  createEffect(
    on(
      () => props.cacheKey,
      (key, prevKey) => {
        if (prevKey !== undefined) {
          outputCache.set(prevKey, output());
        }
        setOutput(key ? (outputCache.get(key) ?? null) : null);
      },
    ),
  );

  // Keep cache in sync after each execution
  createEffect(() => {
    const result = output();
    const key = props.cacheKey;
    if (key && result) {
      outputCache.set(key, result);
    }
  });

  const emptyResult = (): ExecutionResult => ({
    stdout: "",
    stderr: "",
    error: null,
    execution_time_ms: 0,
    metrics: {},
    plots: [],
    tensors: [],
  });

  const handleRun = async (code: string) => {
    setRunning(true);
    setOutput(null);

    if (streaming()) {
      setOutput(emptyResult());
      try {
        await executeStream(code, props.kataId ?? "unknown", {
          onStdout: (line) => {
            setOutput((prev) =>
              prev ? { ...prev, stdout: prev.stdout + line + "\n" } : prev,
            );
          },
          onStderr: (line) => {
            setOutput((prev) =>
              prev
                ? { ...prev, stderr: prev.stderr + line + "\n" }
                : prev,
            );
          },
          onPlot: (raw) => {
            const match = raw.match(PLOT_RE);
            if (match) {
              const plot: PlotData = {
                index: parseInt(match[1]),
                data: `data:image/png;base64,${match[2]}`,
                format: "png",
              };
              setOutput((prev) =>
                prev ? { ...prev, plots: [...prev.plots, plot] } : prev,
              );
            }
          },
          onMetric: (raw) => {
            const match = raw.match(METRIC_RE);
            if (match) {
              const val = parseFloat(match[2]);
              setOutput((prev) =>
                prev
                  ? {
                      ...prev,
                      metrics: {
                        ...prev.metrics,
                        [match[1]]: isNaN(val) ? match[2] : val,
                      },
                    }
                  : prev,
              );
            }
          },
          onTensor: (raw) => {
            const match = raw.match(TENSOR_RE);
            if (match) {
              try {
                const tensor: TensorData = JSON.parse(match[1]);
                setOutput((prev) =>
                  prev
                    ? { ...prev, tensors: [...prev.tensors, tensor] }
                    : prev,
                );
              } catch {
                // skip malformed tensor data
              }
            }
          },
          onDone: (ms) => {
            setOutput((prev) =>
              prev ? { ...prev, execution_time_ms: ms } : prev,
            );
            setRunning(false);
          },
          onError: (msg) => {
            setOutput((prev) =>
              prev ? { ...prev, error: msg } : prev,
            );
            setRunning(false);
          },
        });
      } catch (e) {
        setOutput({
          ...emptyResult(),
          error: e instanceof Error ? e.message : "Stream failed",
        });
        setRunning(false);
      }
    } else {
      try {
        const result = await apiPost<ExecutionResult>("/execute", {
          code,
          kata_id: props.kataId ?? "unknown",
        });
        setOutput(result);
      } catch (e) {
        setOutput({
          ...emptyResult(),
          error: e instanceof Error ? e.message : "Unknown error",
        });
      } finally {
        setRunning(false);
      }
    }
  };

  const toggleMaximize = (panel: "code" | "output") => {
    setMaximized((prev) => (prev === panel ? null : panel));
  };

  return (
    <div class="kata-workspace">
      <Show
        when={maximized() === null}
        fallback={
          <div class="kata-workspace__maximized">
            <Show when={maximized() === "code"}>
              <CodePanel
                maximized={true}
                onToggleMaximize={() => toggleMaximize("code")}
                onRun={handleRun}
                running={running()}
                defaultCode={props.defaultCode}
              />
            </Show>
            <Show when={maximized() === "output"}>
              <OutputPanel
                maximized={true}
                onToggleMaximize={() => toggleMaximize("output")}
                output={output()}
                running={running()}
                streaming={streaming()}
                onToggleStreaming={() => setStreaming((p) => !p)}
              />
            </Show>
          </div>
        }
      >
        <Resizable class="kata-workspace__resizable">
          <Resizable.Panel
            initialSize={0.5}
            minSize={0.2}
            class="kata-workspace__panel"
          >
            <CodePanel
              maximized={false}
              onToggleMaximize={() => toggleMaximize("code")}
              onRun={handleRun}
              running={running()}
              defaultCode={props.defaultCode}
            />
          </Resizable.Panel>
          <Resizable.Handle
            aria-label="Resize code and output panels"
            class="kata-workspace__handle"
          >
            <div class="kata-workspace__handle-bar" />
          </Resizable.Handle>
          <Resizable.Panel
            initialSize={0.5}
            minSize={0.2}
            class="kata-workspace__panel"
          >
            <OutputPanel
              maximized={false}
              onToggleMaximize={() => toggleMaximize("output")}
              output={output()}
              running={running()}
              streaming={streaming()}
              onToggleStreaming={() => setStreaming((p) => !p)}
            />
          </Resizable.Panel>
        </Resizable>
      </Show>
    </div>
  );
}
