import ThemeToggle from "../components/layout/theme-toggle";
import TrackCard from "../components/track-card/track-card";
import { TRACKS } from "../lib/constants";
import { ROUTES } from "../routes";
import "./landing.css";

export default function Landing() {
  return (
    <div class="landing">
      <header class="landing__header">
        <ThemeToggle />
      </header>
      <main class="landing__main">
        <h1 class="landing__title">Python AI Katas</h1>
        <p class="landing__subtitle">
          Learn AI as an engineering discipline, not magic.
          Build intuition through hands-on experimentation.
        </p>
        <div class="landing__tracks">
          <TrackCard
            title={TRACKS.FOUNDATIONAL_AI.name}
            description={TRACKS.FOUNDATIONAL_AI.description}
            status="active"
            href={ROUTES.FOUNDATIONAL_AI}
          />
          <TrackCard
            title={TRACKS.TRADITIONAL_AI_ML.name}
            description={TRACKS.TRADITIONAL_AI_ML.description}
            status="active"
            href={ROUTES.TRADITIONAL_AI_ML}
          />
        </div>
      </main>
    </div>
  );
}
