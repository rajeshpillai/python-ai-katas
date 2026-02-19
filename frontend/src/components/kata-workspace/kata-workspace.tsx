import { createSignal, Show } from "solid-js";
import Resizable from "@corvu/resizable";
import CodePanel from "./code-panel";
import OutputPanel from "./output-panel";
import { apiPost } from "../../lib/api-client";
import "./kata-workspace.css";

type MaximizedPanel = "code" | "output" | null;

interface ExecutionResult {
  stdout: string;
  stderr: string;
  error: string | null;
  execution_time_ms: number;
  metrics: Record<string, unknown>;
  plots: unknown[];
}

interface KataWorkspaceProps {
  kataId?: string;
  defaultCode?: string;
}

export default function KataWorkspace(props: KataWorkspaceProps) {
  const [maximized, setMaximized] = createSignal<MaximizedPanel>(null);
  const [output, setOutput] = createSignal<ExecutionResult | null>(null);
  const [running, setRunning] = createSignal(false);

  const handleRun = async (code: string) => {
    setRunning(true);
    setOutput(null);
    try {
      const result = await apiPost<ExecutionResult>("/execute", {
        code,
        kata_id: props.kataId ?? "unknown",
      });
      setOutput(result);
    } catch (e) {
      setOutput({
        stdout: "",
        stderr: "",
        error: e instanceof Error ? e.message : "Unknown error",
        execution_time_ms: 0,
        metrics: {},
        plots: [],
      });
    } finally {
      setRunning(false);
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
            />
          </Resizable.Panel>
        </Resizable>
      </Show>
    </div>
  );
}
