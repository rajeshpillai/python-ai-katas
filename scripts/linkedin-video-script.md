# LinkedIn Video Script — AI Katas

> A screen-recording script for a **60–75 second** LinkedIn demo of AI Katas.
> Optimized for LinkedIn's vertical feed: shoot at **1080×1920** (9:16) or
> **1920×1080** (16:9) — both autoplay well. Captions are mandatory because
> ~80% of LinkedIn video is watched muted.

---

## Hook line (caption + voice-over)

> **"Learn AI by running it — not reading about it."**

Use this as the opening hard cut and pin it as the post's first line.

---

## Pre-flight checklist

Before hitting record:

- Open the site at `https://rajeshpillai.github.io/python-ai-katas/`
  (or local dev: `npm run dev` + `bash scripts/build-static.sh && python3 -m http.server -d frontend/dist 4173`)
- Pick **dark mode** for visual punch (toggle top-right)
- Browser at 100% zoom, devtools closed, bookmarks bar hidden
- Pre-load these three katas so they're warm:
  - **Foundational AI → Phase 0 → "Visualizing distributions"** (good histogram demo)
  - **Foundational AI → Phase 1 → "Linear regression"** (line + scatter plot)
  - **Traditional AI/ML → Phase 5 → "K-means clustering"** (visually striking)
- Run each one once before recording so Pyodide + numpy + matplotlib are cached
  (first-run loads ~15MB; you don't want that on camera)

---

## Shot list

### Shot 1 — Hook (0:00–0:03)

| | |
|---|---|
| **Screen** | Landing page, full-screen |
| **Action** | Static, slow zoom on the title card |
| **Voice-over** | "Most AI tutorials are videos you watch. This one's a playground you run." |
| **Caption overlay** | `Learn AI by running it.` |

### Shot 2 — Two tracks, your pace (0:03–0:12)

| | |
|---|---|
| **Screen** | Landing → click **Python** card → land on track-picker |
| **Action** | Hover both track cards, point cursor at each |
| **Voice-over** | "Two complementary tracks. Foundational AI builds intuition from first principles. Traditional ML covers the classical stuff people ship." |
| **Caption overlay** | `2 tracks · 125+ katas · pick your own order` |

### Shot 3 — Tutorial section (0:12–0:30)

| | |
|---|---|
| **Screen** | Click **Foundational AI** → kata: *Visualizing distributions* |
| **Action** | Slow scroll through the concept section: heading, intuition paragraph, code-fenced block with syntax highlighting visible |
| **Voice-over** | "Every kata starts with intuition — what problem are we solving and why naive approaches fail. Then a worked example. Then live code." |
| **Caption overlay** | `Concept → Intuition → Code` (cycle through these three on screen) |
| **Hold for 1s** | Pause on the syntax-highlighted Python block so viewers register the colors |

### Shot 4 — Code execution (0:30–0:50)

| | |
|---|---|
| **Screen** | Same kata, scroll to the live editor |
| **Action** | Click into the editor, change a single number (e.g. mean from `0` to `5` or `n_samples` from `100` to `1000`), click **Run** |
| **Voice-over** | "Edit the code right in the page. Hit Run. The plot updates in milliseconds — no install, no setup, no notebook server. It's all running in your browser via Pyodide." |
| **Caption overlay** | `Edit. Run. See it change.` then `Pyodide · 100% browser-side` |
| **B-roll option** | Briefly cut to a second kata (e.g. K-means) and show the cluster plot pop in |

### Shot 5 — Multi-language: Rust (0:50–0:60)

| | |
|---|---|
| **Screen** | Switch language to **Rust** (top-left selector) → open one Rust kata |
| **Action** | Scroll the kata briefly, show Rust code highlighted |
| **Voice-over** | "There's a Rust track too — every algorithm written from scratch, no ML crates. For when you want to see what's actually under the hood." |
| **Caption overlay** | `+ Rust track · algorithms from scratch` |

### Shot 6 — Outro / CTA (0:60–0:72)

| | |
|---|---|
| **Screen** | Cut back to landing page, theme toggle dark→light briefly for visual interest, then center the URL |
| **Action** | Static, large URL on screen |
| **Voice-over** | "Free, open source, runs in your browser. Link in the post." |
| **Caption overlay** | `rajeshpillai.github.io/python-ai-katas` |

---

## On-screen disclosure (last 2 seconds OR pinned in post)

```
Built with LLM assistance · human-reviewed · ongoing review · PRs welcome
```

Either flash this at the end (1s hold) or — preferable for LinkedIn — put it as the closing line of the post body so the video stays clean.

---

## LinkedIn post body (to accompany the video)

> Spent the last few weeks building **AI Katas** — an interactive playground for learning AI from first principles.
>
> 🧩 125+ small, runnable katas
> 🧠 Two tracks: **Foundational AI** (intuition-first) and **Traditional AI/ML** (classical, production-grade)
> 🦀 Bonus **Rust track** with every algorithm written from scratch — no ML crates
> 🌐 Runs entirely in your browser via Pyodide — no install, no signup
>
> Tracks, phases, and katas are independent units. No "correct" order — pick, swap, skip, revisit.
>
> 🔗 Try it: https://rajeshpillai.github.io/python-ai-katas/
> 📦 Code: https://github.com/rajeshpillai/python-ai-katas
>
> *Built with LLM assistance and human-reviewed. More review in progress — corrections and PRs very welcome.*
>
> #MachineLearning #AI #Python #Rust #Education #OpenSource

---

## Recording tips

- **Cursor**: enable a cursor highlighter (e.g. macOS Mousecape, Windows ZoomIt). The cursor draws the eye.
- **Pace**: aim for ~3 cuts every 5 seconds. LinkedIn's feed eats slow video.
- **Silence**: leave 0.5s of dead air between sections — feels professional, gives captions time to register.
- **Music**: low-volume lo-fi or ambient under voice-over (~−18 dB). Skip music if voice-over is strong.
- **Captions**: burn them in — LinkedIn's auto-captions are unreliable. Use Descript, CapCut, or Premiere.
- **Resolution / codec**: export H.264, 1080p, ~10 Mbps. Keep file under 200MB.
- **Length**: hard cap **75 seconds**. The LinkedIn algorithm rewards completion rate; a 60s video watched fully beats a 90s video skipped at 50%.

---

## Optional alternates

If you want a **shorter cut** (≤30s for LinkedIn Shorts / Twitter):

1. 0–3s: Hook caption
2. 3–10s: One kata, scroll tutorial fast
3. 10–22s: Edit code, hit Run, plot updates
4. 22–28s: Mention "two tracks + Rust"
5. 28–30s: URL

If you want a **longer cut** (~2 min for YouTube / blog embed):

- Same 6-shot structure
- Add a **Shot 3.5**: dwell on a metric/tensor visualization (e.g. show `kata_metric` or `kata_tensor` rendering)
- Add a **Shot 5.5**: theme toggle + GitHub icon click → opens repo
