import { createSignal, createEffect } from "solid-js";
import "./code-panel.css";

interface CodePanelProps {
  maximized: boolean;
  onToggleMaximize: () => void;
  onRun: (code: string) => void;
  running: boolean;
  defaultCode?: string;
}

const FALLBACK_CODE = `# Write your Python code here
import numpy as np

x = np.array([1, 2, 3, 4, 5])
print("Data:", x)
print("Mean:", x.mean())
`;

export default function CodePanel(props: CodePanelProps) {
  const initialCode = () => props.defaultCode ?? FALLBACK_CODE;
  const [code, setCode] = createSignal(initialCode());

  createEffect(() => {
    const dc = props.defaultCode;
    if (dc) setCode(dc);
  });

  const handleReset = () => setCode(initialCode());
  const handleRun = () => props.onRun(code());

  return (
    <div class="code-panel">
      <div class="code-panel__header">
        <span class="code-panel__title">Code</span>
        <div class="code-panel__actions">
          <button
            class="code-panel__btn code-panel__btn--run"
            onClick={handleRun}
            disabled={props.running}
          >
            {props.running ? "Running..." : "Run"}
          </button>
          <button class="code-panel__btn" onClick={handleReset}>
            Reset
          </button>
          <button
            class="code-panel__btn code-panel__btn--icon"
            onClick={props.onToggleMaximize}
            aria-label={props.maximized ? "Restore panel" : "Maximize panel"}
            title={props.maximized ? "Restore" : "Maximize"}
          >
            {props.maximized ? "⊡" : "⊞"}
          </button>
        </div>
      </div>
      <textarea
        class="code-panel__editor"
        value={code()}
        onInput={(e) => setCode(e.currentTarget.value)}
        onKeyDown={(e) => {
          if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            handleRun();
          }
        }}
        spellcheck={false}
      />
    </div>
  );
}
