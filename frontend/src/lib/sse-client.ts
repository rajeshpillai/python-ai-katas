import { API_BASE } from "./constants";

export interface SSECallbacks {
  onStdout: (line: string) => void;
  onStderr: (line: string) => void;
  onPlot: (raw: string) => void;
  onMetric: (raw: string) => void;
  onTensor: (raw: string) => void;
  onDone: (executionTimeMs: number) => void;
  onError: (message: string) => void;
}

export async function executeStream(
  code: string,
  kataId: string,
  lang: string,
  callbacks: SSECallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_BASE}/${lang}/execute/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code, kata_id: kataId }),
    signal,
  });

  if (!res.ok || !res.body) {
    throw new Error(`Stream request failed: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";

    for (const chunk of chunks) {
      if (!chunk.trim()) continue;
      const eventMatch = chunk.match(/^event:\s*(\w+)\ndata:\s*(.*)$/s);
      if (!eventMatch) continue;
      const [, type, data] = eventMatch;

      switch (type) {
        case "stdout":
          callbacks.onStdout(data);
          break;
        case "stderr":
          callbacks.onStderr(data);
          break;
        case "plot":
          callbacks.onPlot(data);
          break;
        case "metric":
          callbacks.onMetric(data);
          break;
        case "tensor":
          callbacks.onTensor(data);
          break;
        case "done":
          callbacks.onDone(parseFloat(data));
          break;
        case "error":
          callbacks.onError(data);
          break;
      }
    }
  }
}
