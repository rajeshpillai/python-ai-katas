import { createSignal, Show } from "solid-js";
import Resizable from "@corvu/resizable";
import CodePanel from "./code-panel";
import OutputPanel from "./output-panel";
import "./kata-workspace.css";

type MaximizedPanel = "code" | "output" | null;

export default function KataWorkspace() {
  const [maximized, setMaximized] = createSignal<MaximizedPanel>(null);

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
              />
            </Show>
            <Show when={maximized() === "output"}>
              <OutputPanel
                maximized={true}
                onToggleMaximize={() => toggleMaximize("output")}
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
            />
          </Resizable.Panel>
        </Resizable>
      </Show>
    </div>
  );
}
