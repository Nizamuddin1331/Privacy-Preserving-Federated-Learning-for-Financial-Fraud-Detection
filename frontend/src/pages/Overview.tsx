import { api } from "../services/api";
import { MODE_COLOR, Page, Panel, State, fmt, useApi } from "../components/ui";

const SEES: Record<string, string> = {
  A: "Server holds every raw transaction", B: "Nothing is shared; no collaboration",
  C: "Server sees each bank's raw model update", D: "Server sees updates with calibrated noise",
  E: "Server sees only the masked sum of updates",
};

export default function Overview() {
  const { data, error, loading } = useApi(api.comparison);
  const health = useApi(api.health);
  const top = Math.max(0.01, ...(data ?? []).map(r => r.pr_auc));
  return (
    <Page title="What does privacy cost in fraud detection?"
      lead="Each rung below hides more from the server. The bar shows how much fraud-detection quality (PR-AUC) the bank keeps at that level of protection.">
      <Panel title="The privacy ladder" sub="Latest completed run of each mode on the same test set.">
        <State loading={loading} error={error} empty={!!data && data.length === 0}
          hint="Nothing to show yet. Open Launch runs and choose “Run all modes A–E”." />
        <div className="ladder">
          {(data ?? []).map(r => (
            <div className="rung" key={r.mode}>
              <div className="id" style={{ color: MODE_COLOR[r.mode] }}>{r.mode}</div>
              <div><b>{r.label}</b><div className="sees">{SEES[r.mode]}</div></div>
              <div className="track" role="img" aria-label={`PR-AUC ${fmt(r.pr_auc)}`}><i style={{ width: `${(r.pr_auc / top) * 100}%`, background: MODE_COLOR[r.mode] }} /></div>
              <div>PR-AUC <b>{fmt(r.pr_auc)}</b> · recall {fmt(r.recall, 2)}{r.epsilon != null && <> · ε {fmt(r.epsilon, 1)}</>}</div>
            </div>
          ))}
        </div>
      </Panel>
      <Panel title="System status">
        <State loading={health.loading} error={health.error} />
        {health.data && <p className="sub">API {health.data.status} · PyTorch {health.data.torch} · database {health.data.database} ·
          experiments {Object.entries(health.data.experiments).map(([k, v]) => `${v} ${k}`).join(", ") || "none yet"}</p>}
      </Panel>
    </Page>
  );
}
