import React from "react";
import { createRoot } from "react-dom/client";
import "./twind"; // IMPORTANT: initialize Twind
import App from "./App";

const root = createRoot(document.getElementById("root"));
root.render(<App />);
