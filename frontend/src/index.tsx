import { render } from "solid-js/web";
import { Router, Route, Navigate } from "@solidjs/router";
import App from "./app";
import Landing from "./pages/landing";
import FoundationalAi from "./pages/foundational-ai";
import TraditionalAiMl from "./pages/traditional-ai-ml";
import KataPage from "./pages/kata-page";
import NotFound from "./pages/not-found";
import "./global.css";

const root = document.getElementById("root");

if (!root) {
  throw new Error("Root element not found");
}

render(
  () => (
    <Router root={App}>
      <Route path="/" component={Landing} />
      <Route path="/foundational-ai" component={FoundationalAi}>
        <Route path="/" component={() => <Navigate href="/foundational-ai/0/what-is-data" />} />
        <Route path="/:phaseId/:kataId" component={KataPage} />
      </Route>
      <Route path="/traditional-ai-ml" component={TraditionalAiMl}>
        <Route path="/" component={() => <Navigate href="/traditional-ai-ml/0/what-is-ai" />} />
        <Route path="/:phaseId/:kataId" component={KataPage} />
      </Route>
      <Route path="*" component={NotFound} />
    </Router>
  ),
  root
);
