import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../services/api";
import { Page, Panel, State, Table, useApi } from "../components/ui";

export default function Partitions() {
  const [strategy, setStrategy] = useState("dirichlet");
  const [alpha, setAlpha] = useState(0.5);
  const [n, setN] = useState(5);
  const { data, error, loading, reload } = useApi(() => api.partitions(n, strategy, alpha));
  return (
    <Page title="Bank splits" lead="Real banks do not see the same customers. Preview how the training data is divided, and how uneven it becomes under Non-IID splits.">
      <Panel>
        <div className="form">
          <label>Split<select value={strategy} onChange={e => setStrategy(e.target.value)}><option value="iid">IID</option><option value="dirichlet">Non-IID (Dirichlet)</option><option value="prevalence">Non-IID (fraud prevalence)</option></select></label>
          <label>Alpha (lower = more skew)<input type="number" step={0.1} min={0.1} value={alpha} onChange={e => setAlpha(Number(e.target.value))} /></label>
          <label>Banks<input type="number" min={2} max={20} value={n} onChange={e => setN(Number(e.target.value))} /></label>
        </div>
        <div className="actions"><button onClick={reload}>Preview split</button></div>
      </Panel>
      <State loading={loading} error={error} />
      {data && (<div className="grid2">
        <Panel title="Records and fraud cases per bank">
          <ResponsiveContainer width="100%" height={260}><BarChart data={data.clients}><CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="client" label={{ value: "Bank", position: "insideBottom", offset: -2 }} /><YAxis yAxisId="l" /><YAxis yAxisId="r" orientation="right" /><Tooltip /><Legend />
            <Bar yAxisId="l" dataKey="n" name="Records" fill="#0e7c7b" /><Bar yAxisId="r" dataKey="n_fraud" name="Fraud cases" fill="#b7791f" /></BarChart></ResponsiveContainer>
        </Panel>
        <Panel title="Detail"><Table cols={[{ key: "client", label: "Bank" }, { key: "n", label: "Records", num: true }, { key: "n_fraud", label: "Fraud", num: true },
          { key: "fraud_rate", label: "Fraud rate", num: true, render: r => `${(r.fraud_rate * 100).toFixed(2)}%` }]} rows={data.clients} /></Panel>
      </div>)}
    </Page>
  );
}
