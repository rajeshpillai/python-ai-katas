import { render } from "solid-js/web";
import { Router, Route, Navigate, useParams } from "@solidjs/router";
import App from "./app";
import Landing from "./pages/landing";
import LanguageTracks from "./pages/language-tracks";
import FoundationalAi from "./pages/foundational-ai";
import TraditionalAiMl from "./pages/traditional-ai-ml";
import KataPage from "./pages/kata-page";
import NotFound from "./pages/not-found";
import "./global.css";

const root = document.getElementById("root");

if (!root) {
  throw new Error("Root element not found");
}

function FoundationalAiRedirect() {
  const params = useParams();
  return <Navigate href={`/${params.lang}/foundational-ai/0/what-is-data`} />;
}

function TraditionalAiMlRedirect() {
  const params = useParams();
  return <Navigate href={`/${params.lang}/traditional-ai-ml/0/what-is-ai`} />;
}

render(
  () => (
    <Router root={App}>
      <Route path="/" component={Landing} />
      <Route path="/:lang" component={LanguageTracks} />
      <Route path="/:lang/foundational-ai" component={FoundationalAi}>
        <Route path="/" component={FoundationalAiRedirect} />
        <Route path="/:phaseId/:kataId" component={KataPage} />
      </Route>
      <Route path="/:lang/traditional-ai-ml" component={TraditionalAiMl}>
        <Route path="/" component={TraditionalAiMlRedirect} />
        <Route path="/:phaseId/:kataId" component={KataPage} />
      </Route>
      <Route path="*" component={NotFound} />
    </Router>
  ),
  root
);
