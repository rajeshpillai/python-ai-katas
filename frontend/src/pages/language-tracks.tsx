import { useParams, Navigate } from "@solidjs/router";
import ThemeToggle from "../components/layout/theme-toggle";
import TrackCard from "../components/track-card/track-card";
import { LANGUAGES, TRACKS } from "../lib/constants";
import "./landing.css";

export default function LanguageTracks() {
  const params = useParams<{ lang: string }>();
  const language = () => LANGUAGES.find((l) => l.id === params.lang);

  return (
    <>
      {language() ? (
        <div class="landing">
          <header class="landing__header">
            <ThemeToggle />
          </header>
          <main class="landing__main">
            <h1 class="landing__title">{language()!.name} AI Katas</h1>
            <p class="landing__subtitle">
              Choose a learning track to get started.
            </p>
            <div class="landing__tracks">
              <TrackCard
                title={TRACKS.FOUNDATIONAL_AI.name}
                description={TRACKS.FOUNDATIONAL_AI.description}
                status="active"
                href={`/${params.lang}/foundational-ai`}
              />
              <TrackCard
                title={TRACKS.TRADITIONAL_AI_ML.name}
                description={TRACKS.TRADITIONAL_AI_ML.description}
                status="active"
                href={`/${params.lang}/traditional-ai-ml`}
              />
            </div>
          </main>
        </div>
      ) : (
        <Navigate href="/" />
      )}
    </>
  );
}
