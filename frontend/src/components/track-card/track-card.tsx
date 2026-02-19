import { A } from "@solidjs/router";
import "./track-card.css";

interface TrackCardProps {
  title: string;
  description: string;
  status: "active" | "coming-soon";
  href?: string;
}

export default function TrackCard(props: TrackCardProps) {
  return (
    <>
      {props.status === "active" && props.href ? (
        <A href={props.href} class="track-card track-card--active">
          <div class="track-card__badge track-card__badge--active">Active</div>
          <h2 class="track-card__title">{props.title}</h2>
          <p class="track-card__description">{props.description}</p>
          <div class="track-card__action">Start Learning →</div>
        </A>
      ) : (
        <div class="track-card track-card--disabled">
          <div class="track-card__badge track-card__badge--coming-soon">
            Coming Soon
          </div>
          <h2 class="track-card__title">{props.title}</h2>
          <p class="track-card__description">{props.description}</p>
        </div>
      )}
    </>
  );
}
