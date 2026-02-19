import "./output-panel.css";

interface OutputPanelProps {
  maximized: boolean;
  onToggleMaximize: () => void;
}

export default function OutputPanel(props: OutputPanelProps) {
  return (
    <div class="output-panel">
      <div class="output-panel__header">
        <span class="output-panel__title">Output</span>
        <div class="output-panel__actions">
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
          <p class="output-panel__placeholder">
            Click "Run" to execute your code.
          </p>
        </div>
        <div class="output-panel__viz">
          <p class="output-panel__placeholder">
            Plots and visualizations will appear here.
          </p>
        </div>
      </div>
    </div>
  );
}
