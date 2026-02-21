import { Show, For } from "solid-js";
import type { ExecutionResult } from "../../lib/types";
import MetricsPanel from "./metrics-panel";
import "./output-panel.css";

interface OutputPanelProps {
  maximized: boolean;
  onToggleMaximize: () => void;
  output: ExecutionResult | null;
  running: boolean;
  streaming: boolean;
  onToggleStreaming: () => void;
}

export default function OutputPanel(props: OutputPanelProps) {
  const hasOutput = () => props.output && (props.output.stdout || props.output.stderr || props.output.error);

  return (
    <div class="output-panel">
      <div class="output-panel__header">
        <span class="output-panel__title">Output</span>
        <div class="output-panel__actions">
          <button
            class={`output-panel__btn output-panel__btn--stream ${props.streaming ? "output-panel__btn--active" : ""}`}
            onClick={props.onToggleStreaming}
            title={props.streaming ? "Streaming: ON" : "Streaming: OFF"}
          >
            {props.streaming ? "Stream" : "Batch"}
          </button>
          <Show when={props.output && props.output!.execution_time_ms > 0}>
            <span class="output-panel__time">
              {props.output!.execution_time_ms.toFixed(0)}ms
            </span>
          </Show>
          <button
            class="output-panel__btn output-panel__btn--icon"
            onClick={props.onToggleMaximize}
            aria-label={props.maximized ? "Restore panel" : "Maximize panel"}
            title={props.maximized ? "Restore" : "Maximize"}
          >
            {props.maximized ? "⊡" : "⊞"}
          </button>
        </div>
      </div>
      <div class="output-panel__content">
        <Show when={props.output && Object.keys(props.output!.metrics).length > 0}>
          <MetricsPanel metrics={props.output!.metrics} />
        </Show>
        <div class="output-panel__terminal">
          <Show when={props.running && !props.streaming}>
            <p class="output-panel__running">Running...</p>
          </Show>
          <Show when={!props.running && !props.output}>
            <p class="output-panel__placeholder">
              Click "Run" or press Ctrl+Enter to execute your code.
            </p>
          </Show>
          <Show when={hasOutput()}>
            <Show when={props.output!.stdout}>
              <pre class="output-panel__stdout">{props.output!.stdout}</pre>
            </Show>
            <Show when={props.output!.stderr}>
              <pre class="output-panel__stderr">{props.output!.stderr}</pre>
            </Show>
            <Show when={props.output!.error}>
              <pre class="output-panel__error">{props.output!.error}</pre>
            </Show>
          </Show>
          <Show when={props.running && props.streaming && !hasOutput()}>
            <p class="output-panel__running">Streaming...</p>
          </Show>
        </div>
        <Show when={props.output && props.output!.plots.length > 0}>
          <div class="output-panel__viz">
            <For each={props.output!.plots}>
              {(plot) => (
                <div class="output-panel__plot">
                  <img
                    src={plot.data}
                    alt={`Plot ${plot.index + 1}`}
                    class="output-panel__plot-img"
                  />
                </div>
              )}
            </For>
          </div>
        </Show>
        <Show when={props.output && props.output!.tensors.length > 0}>
          <div class="output-panel__tensors">
            <For each={props.output!.tensors}>
              {(tensor) => (
                <div class="output-panel__tensor">
                  <div class="output-panel__tensor-header">
                    <span class="output-panel__tensor-name">{tensor.name}</span>
                    <span class="output-panel__tensor-shape">
                      [{tensor.shape.join(" x ")}]
                    </span>
                  </div>
                  <div
                    class="output-panel__tensor-grid"
                    style={{
                      "grid-template-columns": `repeat(${tensor.values[0]?.length ?? 1}, 28px)`,
                    }}
                  >
                    <For each={tensor.values.flat()}>
                      {(val) => {
                        const t =
                          tensor.max === tensor.min
                            ? 0.5
                            : (val - tensor.min) / (tensor.max - tensor.min);
                        const r = Math.round(t < 0.5 ? t * 2 * 255 : 255);
                        const g = Math.round(
                          t < 0.5 ? t * 2 * 255 : (1 - t) * 2 * 255,
                        );
                        const b = Math.round(t < 0.5 ? 255 : (1 - t) * 2 * 255);
                        const textColor = t > 0.3 && t < 0.7 ? "#000" : "#fff";
                        return (
                          <div
                            class="output-panel__tensor-cell"
                            style={{
                              background: `rgb(${r}, ${g}, ${b})`,
                              color: textColor,
                            }}
                            title={val.toFixed(4)}
                          >
                            {Math.abs(val) < 10 ? val.toFixed(1) : val.toFixed(0)}
                          </div>
                        );
                      }}
                    </For>
                  </div>
                </div>
              )}
            </For>
          </div>
        </Show>
      </div>
    </div>
  );
}
