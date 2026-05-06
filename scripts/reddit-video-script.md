# Reddit Demo Video — AI Katas

A screen-recording script for a **75–90 second** Reddit-tailored demo of
AI Katas. Reddit rewards *show-don't-tell*: silent + captions + lo-fi
music outperforms voice-over on most subs. Front-load the live code
execution — that's the moment that makes the post.

Companion to [linkedin-video-script.md](linkedin-video-script.md).

---

## Format

| | |
|---|---|
| **Length** | 75–90s |
| **Aspect** | 16:9 · 1920×1080 |
| **Audio** | Lo-fi instrumental (~−24 dB). Voice-over **optional, default off**. |
| **Captions** | Burn-in. Sans-serif, 48pt, white with black stroke. |
| **Encode** | H.264, ≤200MB, ≤5 min |

---

## Pre-flight (do this before recording)

1. Open the **deployed** site, not localhost: `https://rajeshpillai.github.io/python-ai-katas/`
2. **Dark mode on** (top-right toggle) — looks more "developer-tool"
3. Browser at 100% zoom. Hide the bookmarks bar. Close devtools.
4. **Pre-warm** these katas — open each, click Run once, wait for the plot.
   This caches Pyodide + numpy/sklearn/matplotlib so the recording isn't
   blocked by a 15MB cold load:
   - **Foundational AI → Phase 0 → "Visualizing distributions"** (histogram)
   - **Foundational AI → Phase 1 → "Linear regression"** (scatter + line)
   - **Traditional AI/ML → Phase 5 → "K-means clustering"** (cluster plot)
5. Keep one **Rust kata** open in another tab for the cut at 0:55
6. Leave a second tab on the **GitHub repo** in case you want a B-roll cut

> 💡 **Important**: Reddit shows the **first frame** as the post thumbnail. Open mid-demo on a colorful plot, not a logo or landing page.

---

## Hook (post title + first text overlay)

> **"I built an AI learning playground that runs entirely in your browser."**

Same line in both places. The viewer must answer "what is this and should I keep watching" within 2 seconds.

---

## Shot list

### Shot 1 — Cold-open thumbnail (0:00–0:03)

| | |
|---|---|
| **Screen** | Pre-warmed kata page, plot visible, code in the editor |
| **Action** | Static; ~0.5s in, fade-up the caption |
| **Caption** | `I built an AI learning playground` <br> `that runs in your browser.` |
| **Voice-over (optional)** | "Most ML tutorials are videos you watch. This one's a playground you run." |

### Shot 2 — The two tracks (0:03–0:14)

| | |
|---|---|
| **Screen** | Cut to landing page → click **Python** → land on track picker |
| **Action** | Cursor hovers each track card slowly (~1s on each); briefly highlight the kata count |
| **Caption** | `Two tracks. 125+ runnable katas.` <br> *(2s later)* `Pick any order — no fixed path.` |
| **Voice-over (optional)** | "Two tracks: Foundational AI builds intuition from data up. Traditional ML covers the classical stuff people ship. Pick whatever order makes sense to you." |

### Shot 3 — Tutorial structure (0:14–0:30)

| | |
|---|---|
| **Screen** | Click **Foundational AI** → kata: *Visualizing distributions* |
| **Action** | Slow scroll through the kata: heading → intuition paragraph → worked example → syntax-highlighted Python block. Pause ~1s on the code block so the colors register. |
| **Caption** | `Concept` → `Intuition` → `Code` (cycle as you scroll past each) |
| **Voice-over (optional)** | "Each kata starts with intuition — what we're solving and why naive approaches fail. Then a worked example. Then live code." |

### Shot 4 — The money shot (0:30–0:58) ⭐

> **The most important 28 seconds in the video.** Don't rush this.

| | |
|---|---|
| **Screen** | Same kata, scrolled to the editor + Run button |
| **Action 1** | Click into the editor. Change `n_samples = 1000` to `n_samples = 5000`. Click **Run**. Plot updates in <1s. **Hold for 2s** so the viewer registers it. |
| **Action 2** | Change `mu = 0` to `mu = 5`. Click Run. Plot shifts. Hold 2s. |
| **Action 3** | Open a new kata via sidebar (e.g. *Linear regression* or *K-means*). Click Run on the default code. Show a different chart appear. |
| **Caption (sequenced)** | `Edit. Run.` → `No install. No signup. No backend.` → `Pyodide → Python in WebAssembly` |
| **Voice-over (optional)** | "Edit the code. Hit Run. Python compiled to WebAssembly executes it right in the browser. No notebook server, no signup, no install." |

> If you only have time for one variation, do **Action 1** twice (once subtly, once dramatically — e.g. `n_samples = 100` → `n_samples = 50000`) so the speed of the re-render is the takeaway.

### Shot 5 — Rust + theme (0:58–1:12)

| | |
|---|---|
| **Screen** | Switch language toggle (top-left) to **Rust** → open one Rust kata |
| **Action** | Show the Rust kata content + syntax-highlighted Rust code. Then flip the **theme toggle** (top-right) light↔dark for ~1s of visual punch. |
| **Caption** | `Rust track too — algorithms from scratch, no ML crates.` |
| **Voice-over (optional)** | "There's a Rust track too — every algorithm written from scratch, no ML crates. For when you want to see what's actually happening under the hood." |

> 💡 Don't try to *run* the Rust kata in the video — `rustc` isn't in the browser. The Rust track is read-only on the static deploy; the workspace banner already says so. Skip past it.

### Shot 6 — Outro (1:12–1:25)

| | |
|---|---|
| **Screen** | Cut back to the landing page, large URL centered |
| **Action** | Static, slow zoom-in (~5s). End on the URL filling the lower third. |
| **Caption** | `rajeshpillai.github.io/python-ai-katas` <br> `Free · Open source · Comments and PRs welcome` |
| **Voice-over (optional)** | "Free, open source, runs in your browser. Link's in the post. Tear it apart." |

> 💡 Move the LLM disclosure to the post body, not the video. Keeps the close clean.

---

## Reddit post body

> Hey folks — been building **AI Katas**, an interactive playground for learning AI from first principles. Live demo runs entirely in the browser.
>
> What you're looking at:
>
> - 🧩 **125+ small, runnable katas** — each one structured as concept → intuition → code you can edit and run
> - 🧠 **Two tracks**: *Foundational AI* (intuition-first — data → optimization → neural nets → attention → LLMs) and *Traditional AI/ML* (classical — regression, ensembles, time series, RL, productionizing)
> - 🦀 **Bonus Rust track** — every algorithm from scratch, no ML crates
> - 🌐 **Pyodide for execution** — Python in WebAssembly. No install, no signup, hit Run
>
> Tracks/phases/katas are independent units. There's no "correct" sequence — pick, swap, skip, revisit. Buffet, not textbook.
>
> 🔗 Demo: https://rajeshpillai.github.io/python-ai-katas/
> 📦 Source: https://github.com/rajeshpillai/python-ai-katas
>
> **Honest disclosure**: a lot of the code and content was generated with LLM assistance and human-reviewed. More review is in progress, so if you spot anything wrong, hand-wavy, or just bad — issues and PRs are *very* welcome. Roast it.

---

## Subreddit-specific tweaks

| Sub | Title | Lead frame | Emphasize |
|---|---|---|---|
| **r/learnmachinelearning** | *"Browser-based AI/ML playground — looking for feedback from learners"* | A kata showing Concept → Intuition → Code | Two-track choice + non-linear path |
| **r/Python** | *"AI learning app that runs Python entirely in the browser via Pyodide"* | Code editor + plot output | Pyodide + "every kata is a hackable Python script" |
| **r/rust** | *"AI/ML katas with a Rust track — every algorithm from scratch, no ML crates"* | A Rust kata with highlighted code | Lead with Rust; clarify that Rust katas don't run in the browser (no rustc in WASM) |
| **r/SideProject** | *"AI Katas — a free, browser-based AI learning playground"* | Clean landing page | Tech stack (SolidJS + Pyodide + FastAPI) and build details |
| **r/MachineLearning** | ⚠️ Read the rules first. Likely **[D] Discussion** flair, framed as a *teaching resource*, not a project showcase. | — | The educational thesis (intuition-first vs traditional) |

---

## Recording rules of thumb

- **Skip voice-over by default.** Reddit autoplays muted; most viewers never unmute.
- **First frame = thumbnail.** Open mid-demo, not on a logo.
- **No watermarks, no end cards, no "subscribe."** Reddit downvotes these reflexively.
- **Hard cuts only.** No fades, no crossfades. Reddit pacing is fast.
- **No vocal music.** Pure instrumental lo-fi or ambient — vocal music tanks watch-through.
- **Burn captions in.** Reddit's caption layer is unreliable.
- **Cursor highlighter** (Mousecape on macOS, ZoomIt on Windows) — it draws the eye to where the action is.
- **Aim for ~3 cuts every 5 seconds** in the demo sections; 1 cut per 5s in the outro.

---

## Common mistakes to avoid

- **Recording cold.** First Pyodide load is ~15MB. Always pre-warm.
- **Long Run delays.** If a kata takes >2s to render after Run, swap it for a faster one. Time-to-output matters more than which kata it is.
- **Tiny captions.** If you can't read them on a phone screen at arm's length, they're too small.
- **Showing the "Loading torch..." message.** Skip torch katas in the demo unless you've pre-loaded torch in that browser session.
- **Demoing the Run button on a Rust kata.** It's hidden by design — don't try, just scroll past.

---

## Alternates

### 30-second cut (r/coolgithubprojects, r/InternetIsBeautiful, Twitter)

1. 0–2s — Hook caption over a live plot
2. 2–10s — Edit code, hit Run, plot updates (the money shot, no setup)
3. 10–18s — Quick scroll showing two tracks + Rust mention
4. 18–25s — Theme flip + URL on screen
5. 25–30s — LLM disclosure caption

### 3-minute cut (blog embed, YouTube)

- Same 6-shot structure
- After Shot 4: dwell ~15s on the output panel (plots, metrics, tensors)
- After Shot 5: cut to the GitHub repo, scroll the README, briefly show `scripts/build-static.sh` running in a terminal — appeals to "how is this built" curiosity
