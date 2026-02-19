import { render } from "solid-js/web";
import { Router } from "@solidjs/router";
import App from "./app";
import "./global.css";

const root = document.getElementById("root");

if (!root) {
  throw new Error("Root element not found");
}

render(() => <Router root={App} />, root);
