import { createSignal, createResource, For, Show } from "solid-js";
import { A, useParams } from "@solidjs/router";
import { apiGet } from "../../lib/api-client";
import "./sidebar.css";

interface KataItem {
  id: string;
  title: string;
  phase: number;
  sequence: number;
}

interface KatasResponse {
  katas: KataItem[];
  phases: Record<string, string>;
}

interface SidebarProps {
  trackId: string;
}

export default function Sidebar(props: SidebarProps) {
  const [expanded, setExpanded] = createSignal(true);
  const [collapsedPhases, setCollapsedPhases] = createSignal<Set<number>>(
    new Set()
  );
  const params = useParams();

  const [data] = createResource(
    () => props.trackId,
    (trackId) => apiGet<KatasResponse>(`/tracks/${trackId}/katas`)
  );

  const trackTitle = () => {
    const id = props.trackId;
    if (id === "foundational-ai") return "Foundational AI";
    if (id === "traditional-ai-ml") return "Traditional AI/ML";
    return id;
  };

  const phaseNames = () => data()?.phases ?? {};

  const phases = () => {
    const katas = data()?.katas ?? [];
    const grouped = new Map<number, KataItem[]>();
    for (const kata of katas) {
      if (!grouped.has(kata.phase)) grouped.set(kata.phase, []);
      grouped.get(kata.phase)!.push(kata);
    }
    return Array.from(grouped.entries()).sort(([a], [b]) => a - b);
  };

  const togglePhase = (phase: number) => {
    setCollapsedPhases((prev) => {
      const next = new Set(prev);
      if (next.has(phase)) next.delete(phase);
      else next.add(phase);
      return next;
    });
  };

  return (
    <aside class={`sidebar ${expanded() ? "sidebar--expanded" : "sidebar--collapsed"}`}>
      <div class="sidebar__header">
        <button
          class="sidebar__toggle"
          onClick={() => setExpanded(!expanded())}
          aria-label={expanded() ? "Collapse sidebar" : "Expand sidebar"}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="3" y1="6" x2="21" y2="6" />
            <line x1="3" y1="12" x2="21" y2="12" />
            <line x1="3" y1="18" x2="21" y2="18" />
          </svg>
        </button>
        <Show when={expanded()}>
          <span class="sidebar__title">{trackTitle()}</span>
        </Show>
        <A href="/" class="sidebar__home" aria-label="Home">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
            <polyline points="9 22 9 12 15 12 15 22" />
          </svg>
        </A>
      </div>
      <Show when={expanded()}>
        <nav class="sidebar__nav">
          <For each={phases()}>
            {([phase, katas]) => (
              <div class="sidebar__phase">
                <button
                  class="sidebar__phase-header"
                  onClick={() => togglePhase(phase)}
                >
                  <span class="sidebar__phase-arrow">
                    {collapsedPhases().has(phase) ? "▸" : "▾"}
                  </span>
                  <span class="sidebar__phase-label">
                    Phase {phase}
                  </span>
                  <span class="sidebar__phase-name">
                    {phaseNames()[phase] ?? ""}
                  </span>
                </button>
                <Show when={!collapsedPhases().has(phase)}>
                  <ul class="sidebar__kata-list">
                    <For each={katas}>
                      {(kata) => (
                        <li>
                          <A
                            href={`/${props.trackId}/${kata.phase}/${kata.id}`}
                            class="sidebar__kata-link"
                            classList={{
                              "sidebar__kata-link--active":
                                params.kataId === kata.id,
                            }}
                          >
                            <span class="sidebar__kata-seq">
                              {kata.phase}.{kata.sequence}
                            </span>
                            {kata.title}
                          </A>
                        </li>
                      )}
                    </For>
                  </ul>
                </Show>
              </div>
            )}
          </For>
        </nav>
      </Show>
    </aside>
  );
}
