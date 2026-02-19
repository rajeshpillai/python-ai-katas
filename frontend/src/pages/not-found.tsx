import { A } from "@solidjs/router";

export default function NotFound() {
  return (
    <div style={{ "text-align": "center", padding: "80px 24px" }}>
      <h1 style={{ "font-size": "3rem", "margin-bottom": "12px" }}>404</h1>
      <p style={{ color: "var(--text-secondary)", "margin-bottom": "24px" }}>
        Page not found
      </p>
      <A href="/">Back to home</A>
    </div>
  );
}
