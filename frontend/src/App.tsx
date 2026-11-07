import { useEffect, useState } from "react";
import Overview from "./pages/Overview";
import Dataset from "./pages/Dataset";
import Partitions from "./pages/Partitions";
import NewRun from "./pages/NewRun";
import Runs from "./pages/Runs";
import Comparison from "./pages/Comparison";
import Convergence from "./pages/Convergence";
import Privacy from "./pages/Privacy";
import SecureAggregation from "./pages/SecureAggregation";
import Robustness from "./pages/Robustness";

const ROUTES: { id: string; label: string; group: string; el: () => JSX.Element }[] = [
  { id: "overview", label: "Overview", group: "Start", el: Overview },
  { id: "dataset", label: "Transactions", group: "Data", el: Dataset },
  { id: "partitions", label: "Bank splits", group: "Data", el: Partitions },
  { id: "new", label: "Launch runs", group: "Experiments", el: NewRun },
  { id: "runs", label: "Run queue", group: "Experiments", el: Runs },
  { id: "comparison", label: "Mode comparison", group: "Findings", el: Comparison },
  { id: "convergence", label: "Convergence", group: "Findings", el: Convergence },
  { id: "privacy", label: "Privacy budget", group: "Findings", el: Privacy },
  { id: "secagg", label: "Secure aggregation", group: "Findings", el: SecureAggregation },
  { id: "robustness", label: "Poisoning defence", group: "Findings", el: Robustness },
];

const current = () => (location.hash.replace("#/", "") || "overview");

export default function App() {
  const [route, setRoute] = useState(current());
  useEffect(() => { const f = () => setRoute(current()); addEventListener("hashchange", f); return () => removeEventListener("hashchange", f); }, []);
  const active = ROUTES.find(r => r.id === route) ?? ROUTES[0];
  const View = active.el;
  const groups = [...new Set(ROUTES.map(r => r.group))];
  return (
    <div className="shell">
      <aside className="rail">
        <h1>Federated Fraud Lab</h1>
        <p>Banks learn together without sharing transactions.</p>
        <nav>{groups.map(g => (<div key={g}><h2>{g}</h2>
          {ROUTES.filter(r => r.group === g).map(r => <a key={r.id} href={`#/${r.id}`} aria-current={r.id === active.id ? "page" : undefined}>{r.label}</a>)}</div>))}
        </nav>
      </aside>
      <main><View /></main>
    </div>
  );
}
