import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../services/api";
import { Page, Panel, State, Table, fmt, useApi } from "../components/ui";

export default function Privacy() {
  const { data, error, loading } = useApi(api.privacy);
  const sweep = (data ?? []).filter(r => r.tag === "sweep");
  return (
    <Page title="Privacy budget" lead="Differential privacy limits what anyone can learn about a single transaction. A smaller ε means stronger protection, and usually a less accurate model.">
      <State loading={loading} error={error} empty={!!data && data.length === 0} hint="Run the privacy sweep from Launch runs." />
      {sweep.length > 1 && <Panel title="Privacy against quality" sub="Each point is one noise level σ, trained with the same data and rounds. Moving left buys privacy and costs PR-AUC.">
        <ResponsiveContainer width="100%" height={300}><LineChart data={sweep}><CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="epsilon" type="number" scale="log" domain={["auto", "auto"]} tickFormatter={v => v.toFixed(1)} label={{ value: "ε (log scale)", position: "insideBottom", offset: -2 }} />
          <YAxis domain={[0, 1]} /><Tooltip formatter={(v: number) => v.toFixed(3)} labelFormatter={v => `ε = ${Number(v).toFixed(2)}`} /><Legend />
          <Line dataKey="pr_auc" name="PR-AUC" stroke="#5b3fa0" strokeWidth={2} /><Line dataKey="recall" name="Recall" stroke="#b7791f" strokeWidth={2} /></LineChart></ResponsiveContainer>
      </Panel>}
      {data && data.length > 0 && <Panel title="All private runs" sub="ε is the worst case across banks, at the stated δ, for record-level privacy.">
        <Table rows={data} cols={[{ key: "name", label: "Run" }, { key: "noise_multiplier", label: "σ", num: true }, { key: "max_grad_norm", label: "Clip C", num: true },
          { key: "epsilon", label: "ε", num: true, render: r => fmt(r.epsilon, 2) }, { key: "delta", label: "δ", num: true, render: r => r.delta.toExponential(0) },
          { key: "pr_auc", label: "PR-AUC", num: true, render: r => fmt(r.pr_auc) }, { key: "f1", label: "F1", num: true, render: r => fmt(r.f1) }, { key: "recall", label: "Recall", num: true, render: r => fmt(r.recall) }]} />
      </Panel>}
    </Page>
  );
}
