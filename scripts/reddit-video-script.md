# Reddit Demo Video Script — AI Katas

> A screen-recording script for a **75–90 second** Reddit demo of AI Katas.
> Different goals than LinkedIn: Reddit rewards *show, don't tell* — keep
> voice-over **optional** (text overlays + lo-fi music outperform narration
> on most subs), front-load the live code execution, and use a casual,
> "I built this" tone in the post body.
>
> Companion to [linkedin-video-script.md](linkedin-video-script.md). Most
> of the pre-flight setup is shared; differences are flagged below.

---

## Format target

| | |
|---|---|
| **Length** | 75–90s (sweet spot for high-engagement Reddit posts) |
| **Aspect ratio** | 16:9 (1920×1080) — Reddit is desktop-first |
| **Audio** | Lo-fi/ambient music only by default; voice-over optional |
| **Captions** | Required — burn-in, large enough for mobile |
| **File** | MP4 / H.264, ≤200MB, ≤5 min (Reddit's hard limits) |

---

## Hook line (caption, first 2 seconds)

> **"I built an AI learning playground that runs entirely in your browser."**

This is the post title and the opening text overlay. Reddit decides whether
to keep watching in **2 seconds** — the hook *must* answer "what is this and
why should I care."

---

## Pre-flight checklist

Same as the LinkedIn script — see
[linkedin-video-script.md § Pre-flight](linkedin-video-script.md#pre-flight-checklist).
The critical one: pre-load and run each demo kata once so Pyodide and its
packages are warm in the browser cache. **Do not record a cold first run** —
the ~15MB Pyodide download will tank pacing.

Reddit-specific additions:

- Pick **dark mode** — looks more "developer-tool" on Reddit feeds
- Have a second tab ready with the **GitHub repo** in case you want a B-roll cut to the README
- Test on the deployed URL, not localhost — viewers who click through expect the live site to behave the same

---

## Shot list (silent-with-captions cut)

> All voice-over lines below are **optional** — most Reddit posts perform
> equally well or better with caption-only versions and ambient music.

### Shot 1 — Hook (0:00–0:03)

| | |
|---|---|
| **Screen** | Cold open on a kata page mid-demo: a matplotlib plot rendering live, code visible in the editor |
| **Action** | Static, hold |
| **Caption (large, centered)** | `I built an AI learning playground` <br> `that runs in your browser.` |
| **Voice-over (optional)** | "Most ML tutorials are videos. This one's a playground." |

> 💡 **Why open mid-demo?** Reddit feeds preview the first frame as the thumbnail. A live plot + code is more arresting than a landing page logo.

### Shot 2 — The setup (0:03–0:12)

| | |
|---|---|
| **Screen** | Cut to landing page → click into Python card → show two-track picker |
| **Action** | Cursor moves between **Foundational AI** and **Traditional AI/ML** cards, hovering each |
| **Caption** | `Two tracks: intuition-first OR classical ML` <br> *(then 3s later)* `125+ runnable katas. Pick any order.` |
| **Voice-over (optional)** | "Two tracks. Foundational AI builds intuition. Traditional ML covers the classical stuff. Pick whatever order makes sense to you." |

### Shot 3 — Tutorial section (0:12–0:28)

| | |
|---|---|
| **Screen** | Click **Foundational AI** → kata: *Visualizing distributions* (or whichever you pre-warmed) |
| **Action** | Smooth scroll through the kata content: heading → intuition paragraph → worked example → syntax-highlighted Python code block |
| **Caption** | `Concept` *(at top of scroll)* → `Intuition` *(mid)* → `Code` *(at the editor)* — three quick text pops as you scroll past each section |
| **Voice-over (optional)** | "Each kata starts with intuition — what we're solving and why naive approaches fail. Then the code." |

### Shot 4 — The money shot (0:28–0:55)

> **The most important shot in the video.** This is where Reddit's "wait
> what, that runs in the browser?" moment happens. Spend the most time
> here. Don't rush.

| | |
|---|---|
| **Screen** | Same kata, scroll to the editor |
| **Action** | Click into the editor, change a parameter (e.g. `n_samples` from `100` to `2000`, or mean from `0` to `5`), click **Run**. Plot updates within a second. **Do this twice.** Once to demonstrate, once to show speed. |
| **Caption** | `No install. No signup. No backend.` <br> *(after first run)* `Pyodide → Python in WebAssembly` |
| **Voice-over (optional)** | "Edit the code. Hit Run. Python compiled to WebAssembly executes it right in the browser. No install, no signup, no notebook server." |

> 💡 **Bonus shot**: cut briefly to a second kata (K-means or convolution) and run it — variety sells the platform.

### Shot 5 — The Rust track + theme toggle (0:55–1:10)

| | |
|---|---|
| **Screen** | Switch language to **Rust** (top-left) → open one Rust kata |
| **Action** | Show Rust syntax highlighting briefly, then flip the **theme toggle** (top-right) light↔dark for visual punch |
| **Caption** | `Rust track too — algorithms from scratch, no ML crates` |
| **Voice-over (optional)** | "There's a Rust track too — every algorithm written from scratch, no ML crates. For when you want to see what's actually happening under the hood." |

### Shot 6 — Outro / CTA (1:10–1:25)

| | |
|---|---|
| **Screen** | Cut back to landing page; center the URL in big text |
| **Action** | Hold for ~5s; subtle zoom-in on the URL |
| **Caption** | `rajeshpillai.github.io/python-ai-katas` <br> *(smaller, beneath)* `Free · Open source · Comments and PRs welcome` |
| **Voice-over (optional)** | "Free, open source, runs in your browser. Link's in the post. Tear it apart." |

### Shot 7 — Disclosure (1:25–1:30, optional)

| | |
|---|---|
| **Screen** | Plain background or very faded landing page |
| **Caption** | `Built with LLM assistance · human-reviewed · ongoing review` |

> 💡 **Or skip Shot 7** and move the disclosure to the post body — keeps the video clean and ends on the URL.

---

## Reddit post body (paste in the comment field)

> Hey folks — been tinkering on **AI Katas**, an interactive playground for learning AI from first principles. Live demo runs entirely in your browser.
>
> The shape:
>
> - 🧩 **125+ small, runnable katas** — concept, intuition, then live code you can edit and run
> - 🧠 **Two tracks**: *Foundational AI* (intuition-first — data → optimization → neural nets → attention → LLMs) and *Traditional AI/ML* (classical — regression, ensembles, time series, RL, productionizing)
> - 🦀 **Bonus Rust track** — every algorithm from scratch, no ML crates
> - 🌐 **Pyodide for execution** — no install, no signup, just edit code and hit Run
>
> Tracks/phases/katas are independent units. There's no "correct" sequence — pick, swap, skip, revisit. Treat it like a buffet, not a textbook.
>
> 🔗 Demo: https://rajeshpillai.github.io/python-ai-katas/
> 📦 Source: https://github.com/rajeshpillai/python-ai-katas
>
> **Honest disclosure**: a lot of the code and content was generated with LLM assistance and human-reviewed. More review is in progress, so if you spot anything wrong, hand-wavy, or just plain bad — issues and PRs are *very* welcome. Roast it.

---

## Subreddit-specific tweaks

Reddit is not one audience. Tune the title, the leading frame of the video, and the post-body emphasis per sub:

### r/learnmachinelearning

- **Title**: "Built an interactive AI/ML playground (browser-based, no install) — looking for feedback from learners"
- **Lead frame**: a kata showing the *Concept → Intuition → Code* structure
- **Post emphasis**: the two-track choice + non-linear path (most of this audience is overwhelmed by curriculum overload)

### r/Python

- **Title**: "Built an AI learning app that runs Python entirely in your browser via Pyodide — no install"
- **Lead frame**: code editor + plot output, *not* the landing page
- **Post emphasis**: the Pyodide tech + "every kata is a runnable Python script you can hack on"

### r/rust

- **Title**: "AI/ML katas with a Rust track — every algorithm from scratch, no ML crates"
- **Lead frame**: a Rust kata with syntax-highlighted code
- **Post emphasis**: lead with the Rust track, mention Python as a sibling
- **Note**: Rust katas don't run in the browser (no `rustc` in WASM); call this out directly so the audience isn't surprised

### r/SideProject

- **Title**: "AI Katas — a free, browser-based AI learning playground I've been building"
- **Lead frame**: a clean landing page shot with the title visible
- **Post emphasis**: the journey/tech stack (SolidJS + Pyodide + FastAPI) — this sub loves build details
- Optional: a brief "what I learned" / "what's next" section

### r/MachineLearning

- ⚠️ **Read the rules first.** This sub has strict self-promotion rules. Likely best in the **[D] Discussion** flair as a teaching-resource share, not a project showcase. Lead with the *educational thesis* (intuition-first vs traditional), not the demo.

---

## Recording tips (Reddit-specific)

- **Skip voice-over by default**. Reddit autoplays muted; most viewers never unmute. Captions + ambient lo-fi (volume ~−24 dB) outperform a polished voice-over for engagement.
- **First frame matters more than first 3 seconds** — Reddit shows it as the thumbnail. Open mid-demo, not on a logo.
- **No watermarks, no end cards, no calls-to-subscribe** — Reddit hates these and will downvote. Outro is the URL only.
- **Caption font**: sans-serif, white with black stroke, ~48pt. Burn in — Reddit's caption layer is unreliable.
- **Resolution**: 1080p. Lower won't matter (Reddit recompresses), higher wastes bandwidth.
- **No music with vocals** — gets stuck in the recommendation algorithm and tanks watch-through. Pure instrumental lo-fi or ambient.
- **Cut hard, not soft.** No fades. Reddit pacing is fast.

---

## Optional alternates

### 30-second cut (for r/coolgithubprojects, r/InternetIsBeautiful)

1. 0–2s: Hook caption over a live plot
2. 2–10s: Edit code → Run → plot updates (the money shot, no setup)
3. 10–18s: Quick scroll showing two tracks + Rust
4. 18–25s: Theme toggle + URL
5. 25–30s: Disclosure caption

### 3-minute cut (for blog embed or longer demo)

- Same 7-shot structure
- Add a **Shot 4.5**: dwell on the output panel (plots, metrics, tensors) for 15s
- Add a **Shot 5.5**: open the GitHub repo, scroll the README, show the static-build script briefly — appeals to the "how is this built" curiosity
