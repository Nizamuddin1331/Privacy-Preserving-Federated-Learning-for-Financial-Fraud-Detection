import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../services/api";
import { Page, Panel, State, Table, fmt, useApi } from "../components/ui";

export default function Robustness() {
  const { data, error, loading } = useApi(api.robustness);
  const label = (r: { aggregator: string; secure: boolean }) => (r.secure ? "fedavg + secagg" : r.aggregator);
  const attacks = [...new Set((data ?? []).filter(r => r.attack !== "none").map(r => r.attack))];
  const aggs = [...new Set((data ?? []).filter(r => r.attack !== "none").map(label))];
  const chart = attacks.map(a => ({ attack: a, ...Object.fromEntries(aggs.map(g => [g, (data ?? []).find(r => r.attack === a && label(r) === g)?.pr_auc ?? 0])) }));
  const clean = (data ?? []).find(r => r.attack === "none");
  const colors = ["#b42318", "#0e7c7b", "#5b3fa0", "#b7791f"];
  return (
    <Page title="Poisoning defence" lead="One dishonest bank can steer a shared model. Robust aggregation rules resist it, but they need to see individual updates, which secure aggregation hides.">
      <State loading={loading} error={error} empty={!!data && data.length === 0} hint="Run the poisoning grid from Launch runs." />
      {chart.length > 0 && <Panel title="PR-AUC under attack" sub={clean ? `Without any attack the model reaches ${fmt(clean.pr_auc)}.` : undefined}>
        <ResponsiveContainer width="100%" height={300}><BarChart data={chart}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="attack" /><YAxis domain={[0, 1]} /><Tooltip formatter={(v: number) => v.toFixed(3)} /><Legend />
          {aggs.map((g, i) => <Bar key={g} dataKey={g} fill={colors[i % colors.length]} />)}</BarChart></ResponsiveContainer>
      </Panel>}
      {data && data.length > 0 && <Panel title="All runs">
        <Table rows={data} cols={[{ key: "attack", label: "Attack" }, { key: "n_malicious", label: "Dishonest banks", num: true }, { key: "agg", label: "Aggregation", render: r => label(r) },
          { key: "pr_auc", label: "PR-AUC", num: true, render: r => fmt(r.pr_auc) }, { key: "f1", label: "F1", num: true, render: r => fmt(r.f1) }, { key: "recall", label: "Recall", num: true, render: r => fmt(r.recall) }]} />
      </Panel>}
    </Page>
  );
}
