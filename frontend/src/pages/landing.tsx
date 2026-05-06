import { For } from "solid-js";
import ThemeToggle from "../components/layout/theme-toggle";
import RepoLink from "../components/layout/repo-link";
import Footer from "../components/layout/footer";
import TrackCard from "../components/track-card/track-card";
import { LANGUAGES } from "../lib/constants";
import "./landing.css";

export default function Landing() {
  return (
    <div class="landing">
      <header class="landing__header">
        <RepoLink />
        <ThemeToggle />
      </header>
      <main class="landing__main">
        <h1 class="landing__title">AI Katas</h1>
        <p class="landing__subtitle">
          Learn AI as an engineering discipline, not magic.
          Build intuition through hands-on experimentation.
        </p>
        <div class="landing__tracks">
          <For each={LANGUAGES}>
            {(lang) => (
              <TrackCard
                title={lang.name}
                description={lang.description}
                status="active"
                href={`/${lang.id}`}
              />
            )}
          </For>
        </div>
      </main>
      <Footer />
    </div>
  );
}
