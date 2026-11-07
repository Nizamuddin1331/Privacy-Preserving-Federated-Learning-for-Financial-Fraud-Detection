import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../services/api";
import { MODE_COLOR, Page, Panel, State, useApi } from "../components/ui";

export default function Convergence() {
  const { data, error, loading } = useApi(api.convergence);
  const rounds = Math.max(0, ...(data ?? []).flatMap(s => s.history.map(h => h.round)));
  const rows = (key: "pr_auc" | "train_loss" | "epsilon") => Array.from({ length: rounds }, (_, i) => ({
    round: i + 1, ...Object.fromEntries((data ?? []).map(s => [s.mode, s.history[i]?.[key] ?? null])) }));
  const chart = (key: "pr_auc" | "train_loss" | "epsilon", modes?: string[]) => (
    <ResponsiveContainer width="100%" height={280}><LineChart data={rows(key)}><CartesianGrid strokeDasharray="3 3" vertical={false} />
      <XAxis dataKey="round" label={{ value: "Round", position: "insideBottom", offset: -2 }} /><YAxis /><Tooltip formatter={(v: number) => v?.toFixed(3)} /><Legend />
      {(data ?? []).filter(s => !modes || modes.includes(s.mode)).map(s => <Line key={s.mode} type="monotone" dataKey={s.mode} name={`${s.mode}: ${s.label}`} stroke={MODE_COLOR[s.mode]} strokeWidth={2} dot={false} connectNulls />)}
    </LineChart></ResponsiveContainer>);
  return (
    <Page title="Convergence" lead="How quickly each mode learns. For centralised and local-only training a “round” is a block of local epochs, so the curves are comparable.">
      <State loading={loading} error={error} empty={!!data && data.length === 0} hint="Run all modes A–E first." />
      {data && data.length > 0 && (<div className="grid2">
        <Panel title="PR-AUC per round">{chart("pr_auc")}</Panel>
        <Panel title="Training loss per round">{chart("train_loss")}</Panel>
        <Panel title="Privacy budget spent (ε)" sub="Grows with every round for modes D and E; lower is stronger privacy.">{chart("epsilon", ["D", "E"])}</Panel>
      </div>)}
    </Page>
  );
}
