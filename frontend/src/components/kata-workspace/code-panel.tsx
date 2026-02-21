import { createSignal, createMemo, createEffect, Show } from "solid-js";
import { parseSliderConfigs, applySliderValues } from "../../lib/slider-config";
import { createCodeMirror } from "../../lib/create-codemirror";
import SliderBar from "./slider-bar";
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
  const [sliderValues, setSliderValues] = createSignal<Record<string, number>>(
    {},
  );

  const handleRun = () => {
    const configs = sliderConfigs();
    const finalCode =
      configs.length > 0
        ? applySliderValues(code(), configs, sliderValues())
        : code();
    props.onRun(finalCode);
  };

  const { ref: editorRef, setCode: setEditorCode } = createCodeMirror({
    code,
    onCodeChange: setCode,
    onCtrlEnter: handleRun,
  });

  createEffect(() => {
    const dc = props.defaultCode;
    if (dc) {
      setCode(dc);
      setEditorCode(dc);
      setSliderValues({});
    }
  });

  const sliderConfigs = createMemo(() => parseSliderConfigs(code()));

  const handleReset = () => {
    const ic = initialCode();
    setCode(ic);
    setEditorCode(ic);
    setSliderValues({});
  };

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
      <Show when={sliderConfigs().length > 0}>
        <SliderBar
          configs={sliderConfigs()}
          values={sliderValues()}
          onChange={(name, value) =>
            setSliderValues((prev) => ({ ...prev, [name]: value }))
          }
        />
      </Show>
      <div ref={editorRef} class="code-panel__editor" />
    </div>
  );
}
