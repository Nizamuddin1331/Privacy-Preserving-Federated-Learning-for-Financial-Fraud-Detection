import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../services/api";
import { ConfusionMatrix, MODE_COLOR, Page, Panel, State, Table, fmt, useApi } from "../components/ui";

export default function Comparison() {
  const { data, error, loading } = useApi(api.comparison);
  const metrics = ["precision", "recall", "f1", "pr_auc"];
  const chart = metrics.map(m => ({ metric: m === "pr_auc" ? "PR-AUC" : m[0].toUpperCase() + m.slice(1), ...Object.fromEntries((data ?? []).map(r => [r.mode, (r as any)[m]])) }));
  return (
    <Page title="Mode comparison" lead="Centralised training is the ceiling, local-only is the floor. The question is how close privacy-preserving federation gets to the ceiling.">
      <State loading={loading} error={error} empty={!!data && data.length === 0} hint="Run all modes A–E first." />
      {data && data.length > 0 && (<>
        <Panel title="Quality on the held-out test set" sub="PR-AUC is the headline number: with so few frauds, plain accuracy would reward a model that never flags anything.">
          <ResponsiveContainer width="100%" height={300}><BarChart data={chart}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="metric" /><YAxis domain={[0, 1]} /><Tooltip formatter={(v: number) => v.toFixed(3)} /><Legend />
            {data.map(r => <Bar key={r.mode} dataKey={r.mode} name={`${r.mode}: ${r.label}`} fill={MODE_COLOR[r.mode]} />)}</BarChart></ResponsiveContainer>
        </Panel>
        <Panel title="Full results">
          <Table rows={data} cols={[{ key: "label", label: "Mode", render: r => `${r.mode}: ${r.label}` },
            ...["precision", "recall", "f1", "pr_auc", "roc_auc"].map(k => ({ key: k, label: k.replace("_", "-").toUpperCase(), num: true, render: (r: any) => fmt(r[k]) })),
            { key: "fpr", label: "FPR", num: true, render: r => fmt(r.fpr, 4) }, { key: "epsilon", label: "ε", num: true, render: r => fmt(r.epsilon, 2) },
            { key: "uplink_mb", label: "Uplink MB", num: true, render: r => fmt(r.uplink_mb, 2) }]} />
        </Panel>
        <Panel title="What the fraud team would see" sub="Missed fraud costs money; false alarms cost customer trust.">
          <div style={{ display: "flex", flexWrap: "wrap", gap: 28 }}>{data.map(r => <div key={r.mode}><p style={{ margin: "0 0 6px" }}><b>{r.mode}: {r.label}</b></p><ConfusionMatrix cm={r.confusion_matrix} /></div>)}</div>
        </Panel>
      </>)}
    </Page>
  );
}
