import { createSignal, createResource, createMemo, Show } from "solid-js";
import { useParams } from "@solidjs/router";
import KataWorkspace from "../components/kata-workspace/kata-workspace";
import MarkdownContent from "../components/markdown-content/markdown-content";
import { apiGetText } from "../lib/api-client";
import { PHASE_NAMES } from "../lib/constants";
import "./kata-page.css";

type Tab = "concept" | "code";

function extractCodeFromMarkdown(md: string): string | undefined {
  const liveCodeIdx = md.indexOf("## Live Code");
  if (liveCodeIdx === -1) return undefined;
  const after = md.slice(liveCodeIdx);
  const codeBlockMatch = after.match(/```python\n([\s\S]*?)```/);
  return codeBlockMatch ? codeBlockMatch[1].trimEnd() + "\n" : undefined;
}

export default function KataPage() {
  const params = useParams();
  const [activeTab, setActiveTab] = createSignal<Tab>("concept");

  const phaseNum = () => parseInt(params.phaseId, 10);
  const phaseName = () => PHASE_NAMES[phaseNum()] ?? "";
  const kataId = () => params.kataId?.replace(/-/g, " ") ?? "";

  const [content] = createResource(
    () => ({ trackId: "foundational-ai", phaseId: params.phaseId, kataId: params.kataId }),
    (source) =>
      apiGetText(
        `/tracks/${source.trackId}/katas/${source.phaseId}/${source.kataId}/content`
      ).catch(() => null)
  );

  const starterCode = createMemo(() => {
    const md = content();
    if (!md) return undefined;
    return extractCodeFromMarkdown(md);
  });

  return (
    <div class="kata-page">
      <div class="kata-page__header">
        <div class="kata-page__info">
          <span class="kata-page__phase">
            Phase {params.phaseId} — {phaseName()}
          </span>
          <h1 class="kata-page__title">{kataId()}</h1>
        </div>
        <div class="kata-page__tabs">
          <button
            class="kata-page__tab"
            classList={{ "kata-page__tab--active": activeTab() === "concept" }}
            onClick={() => setActiveTab("concept")}
          >
            Concept
          </button>
          <button
            class="kata-page__tab"
            classList={{ "kata-page__tab--active": activeTab() === "code" }}
            onClick={() => setActiveTab("code")}
          >
            Code
          </button>
        </div>
      </div>
      <div class="kata-page__content">
        <Show when={activeTab() === "concept"}>
          <div class="kata-page__concept">
            <Show when={content.loading}>
              <p class="kata-page__loading">Loading content...</p>
            </Show>
            <Show when={!content.loading && content()}>
              <MarkdownContent source={content()!} />
            </Show>
            <Show when={!content.loading && !content()}>
              <p class="kata-page__empty">
                Content for this kata is not yet available.
              </p>
            </Show>
          </div>
        </Show>
        <Show when={activeTab() === "code"}>
          <KataWorkspace kataId={params.kataId} defaultCode={starterCode()} />
        </Show>
      </div>
    </div>
  );
}
