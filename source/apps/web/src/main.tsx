import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { StatusCheckCoordinator } from "./modelStatus";
import "./styles/tokens.css";
import "./styles/app.css";

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("Root element #root is missing in index.html");
}

// One coordinator per app lifetime, created OUTSIDE the StrictMode boundary
// (review 0036 D1B-IF-03): the automatic initial status check survives
// StrictMode's effect replay and runs exactly once; a fresh page load gets a
// fresh coordinator and checks again.
const statusCoordinator = new StatusCheckCoordinator();

createRoot(rootElement).render(
  <StrictMode>
    <App statusCoordinator={statusCoordinator} />
  </StrictMode>,
);
