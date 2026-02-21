import { Show, For } from "solid-js";
import "./metrics-panel.css";

interface MetricsPanelProps {
  metrics: Record<string, number | string>;
}

export default function MetricsPanel(props: MetricsPanelProps) {
  const entries = () => Object.entries(props.metrics);

  return (
    <Show when={entries().length > 0}>
      <div class="metrics-panel">
        <For each={entries()}>
          {([key, value]) => (
            <div class="metrics-panel__item">
              <span class="metrics-panel__label">{key.replace(/_/g, " ")}</span>
              <span class="metrics-panel__value">
                {typeof value === "number" ? value.toFixed(4) : String(value)}
              </span>
            </div>
          )}
        </For>
      </div>
    </Show>
  );
}
