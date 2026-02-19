import type { RouteSectionProps } from "@solidjs/router";
import { Route } from "@solidjs/router";
import { ThemeProvider } from "./context/theme-context";
import Landing from "./pages/landing";
import FoundationalAi from "./pages/foundational-ai";
import KataPage from "./pages/kata-page";
import NotFound from "./pages/not-found";

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

export default function App(props: RouteSectionProps) {
  return (
    <ThemeProvider>
      <Route path="/" component={Landing} />
      <Route path="/foundational-ai" component={FoundationalAi}>
        <Route path="/" component={Welcome} />
        <Route path="/:phaseId/:kataId" component={KataPage} />
      </Route>
      <Route path="*" component={NotFound} />
      {props.children}
    </ThemeProvider>
  );
}
