import { API_BASE } from "./constants";
import { isStaticBuild, tryStaticGet, tryStaticGetText } from "./static-content";

function base(lang: string) {
  return `${API_BASE}/${lang}`;
}

export async function apiGet<T>(path: string, lang: string): Promise<T> {
  if (isStaticBuild()) {
    const handled = tryStaticGet<T>(path);
    if (handled) return handled;
  }
  const url = `${base(lang)}${path}`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`GET ${path} failed: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function apiGetText(path: string, lang: string): Promise<string> {
  if (isStaticBuild()) {
    const handled = tryStaticGetText(path);
    if (handled) return handled;
  }
  const url = `${base(lang)}${path}`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`GET ${path} failed: ${res.status} ${res.statusText}`);
  }
  return res.text();
}

export async function apiPost<T>(path: string, body: unknown, lang: string): Promise<T> {
  // Note: in static builds, POST callers should route to pyodide-runner
  // directly (see executeStream in sse-client.ts). This branch only fires
  // for live-backend dev.
  const url = `${base(lang)}${path}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`POST ${path} failed: ${res.status} ${res.statusText}`);
  }
  return res.json();
}
