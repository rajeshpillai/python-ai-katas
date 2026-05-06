// Static-build replacements for the network calls in api-client.ts.
// Loads bundled kata-index.json + per-kata markdown directly from the asset
// tree produced by scripts/build-static.sh.

const BASE = import.meta.env.BASE_URL || "/";

interface IndexEntry {
  katas: Array<{
    id: string;
    title: string;
    phase: number;
    sequence: number;
    track_id: string;
  }>;
  phases: Record<string, string>;
}

let indexPromise: Promise<Record<string, IndexEntry>> | null = null;

function loadIndex(): Promise<Record<string, IndexEntry>> {
  if (!indexPromise) {
    indexPromise = fetch(`${BASE}katas-index.json`).then((r) => {
      if (!r.ok) throw new Error(`Failed to load kata index: ${r.status}`);
      return r.json();
    });
  }
  return indexPromise;
}

export async function staticGetKatas(trackId: string): Promise<IndexEntry> {
  const index = await loadIndex();
  const entry = index[trackId];
  if (!entry) throw new Error(`Track '${trackId}' not in index`);
  // Phase keys come back as strings from JSON; the existing API returned
  // them as numeric keys but JS object keys are strings either way, so this
  // is shape-compatible with what the frontend already consumes.
  return entry;
}

export async function staticGetKataContent(
  trackId: string,
  phaseId: number,
  kataId: string,
  lang: string,
): Promise<string> {
  const index = await loadIndex();
  const entry = index[trackId];
  if (!entry) throw new Error(`Track '${trackId}' not in index`);
  const kata = entry.katas.find(
    (k) => k.id === kataId && k.phase === phaseId,
  );
  if (!kata) throw new Error(`Kata '${kataId}' not in phase ${phaseId}`);

  const seq = String(kata.sequence).padStart(2, "0");
  const url = `${BASE}content/${lang}/${trackId}/phase-${phaseId}/${seq}-${kataId}.md`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Markdown fetch failed: ${res.status}`);
  return res.text();
}

// Match path shapes from api-client.ts callers.
const KATAS_RE = /^\/tracks\/([^/]+)\/katas$/;
const CONTENT_RE = /^\/tracks\/([^/]+)\/katas\/(\d+)\/([^/]+)\/content$/;

export function isStaticBuild(): boolean {
  return import.meta.env.VITE_STATIC_BUILD === "true";
}

export function tryStaticGet<T>(path: string): Promise<T> | null {
  const katas = path.match(KATAS_RE);
  if (katas) return staticGetKatas(katas[1]) as unknown as Promise<T>;
  return null;
}

export function tryStaticGetText(path: string, lang: string): Promise<string> | null {
  const content = path.match(CONTENT_RE);
  if (content) {
    return staticGetKataContent(content[1], parseInt(content[2], 10), content[3], lang);
  }
  return null;
}
