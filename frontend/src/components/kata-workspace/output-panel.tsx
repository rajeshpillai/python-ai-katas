import { Show } from "solid-js";
import "./output-panel.css";

interface ExecutionResult {
  stdout: string;
  stderr: string;
  error: string | null;
  execution_time_ms: number;
  metrics: Record<string, unknown>;
  plots: unknown[];
}

interface OutputPanelProps {
  maximized: boolean;
  onToggleMaximize: () => void;
  output: ExecutionResult | null;
  running: boolean;
}

export default function OutputPanel(props: OutputPanelProps) {
  return (
    <div class="output-panel">
      <div class="output-panel__header">
        <span class="output-panel__title">Output</span>
        <div class="output-panel__actions">
          <Show when={props.output}>
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
        <div class="output-panel__terminal">
          <Show when={props.running}>
            <p class="output-panel__running">Running...</p>
          </Show>
          <Show when={!props.running && !props.output}>
            <p class="output-panel__placeholder">
              Click "Run" or press Ctrl+Enter to execute your code.
            </p>
          </Show>
          <Show when={!props.running && props.output}>
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
        </div>
      </div>
    </div>
  );
}
