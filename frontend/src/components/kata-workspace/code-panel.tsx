import { createSignal } from "solid-js";
import "./code-panel.css";

interface CodePanelProps {
  maximized: boolean;
  onToggleMaximize: () => void;
}

const DEFAULT_CODE = `# Write your Python code here
import numpy as np

x = np.array([1, 2, 3, 4, 5])
print("Data:", x)
print("Mean:", x.mean())
`;

export default function CodePanel(props: CodePanelProps) {
  const [code, setCode] = createSignal(DEFAULT_CODE);

  const handleReset = () => setCode(DEFAULT_CODE);

  return (
    <div class="code-panel">
      <div class="code-panel__header">
        <span class="code-panel__title">Code</span>
        <div class="code-panel__actions">
          <button class="code-panel__btn code-panel__btn--run">Run</button>
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
        spellcheck={false}
      />
    </div>
  );
}
