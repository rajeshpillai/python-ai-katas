import { render } from "solid-js/web";
import { Router, Route } from "@solidjs/router";
import App from "./app";
import Landing from "./pages/landing";
import FoundationalAi from "./pages/foundational-ai";
import KataPage from "./pages/kata-page";
import NotFound from "./pages/not-found";
import "./global.css";

function Welcome() {
  return (
    <div class="foundational-ai__welcome">
      <h2>Welcome to Foundational AI</h2>
      <p>
        Select a kata from the sidebar to begin your learning journey.
        Start with Phase 0 to build data intuition before diving into models.
      </p>
    </div>
  );
}

const root = document.getElementById("root");

if (!root) {
  throw new Error("Root element not found");
}

render(
  () => (
    <Router root={App}>
      <Route path="/" component={Landing} />
      <Route path="/foundational-ai" component={FoundationalAi}>
        <Route path="/" component={Welcome} />
        <Route path="/:phaseId/:kataId" component={KataPage} />
      </Route>
      <Route path="*" component={NotFound} />
    </Router>
  ),
  root
);
