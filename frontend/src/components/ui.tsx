import { ReactNode, useCallback, useEffect, useState } from "react";

export const MODE_COLOR: Record<string, string> = { A: "#6b7785", B: "#9aa7b3", C: "#0e7c7b", D: "#5b3fa0", E: "#b7791f" };
export const fmt = (v: number | null | undefined, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "–" : v.toFixed(d));

export function useApi<T>(fn: () => Promise<T>, pollMs = 0) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const load = useCallback(() => fn().then(d => { setData(d); setError(null); }).catch(e => setError(String(e.message ?? e))).finally(() => setLoading(false)), []); // eslint-disable-line
  useEffect(() => { load(); if (!pollMs) return; const t = setInterval(load, pollMs); return () => clearInterval(t); }, [load, pollMs]);
  return { data, error, loading, reload: load };
}

export function Page({ title, lead, children }: { title: string; lead: string; children: ReactNode }) {
  return (<><header><h1>{title}</h1><p>{lead}</p></header>{children}</>);
}
export function Panel({ title, sub, children }: { title?: string; sub?: string; children: ReactNode }) {
  return (<section className="panel">{title && <h2>{title}</h2>}{sub && <p className="sub">{sub}</p>}{children}</section>);
}
export function Stat({ label, value }: { label: string; value: ReactNode }) {
  return <div className="stat"><b>{value}</b><span>{label}</span></div>;
}
export function Badge({ status }: { status: string }) { return <span className={`badge ${status}`}>{status}</span>; }

export function State({ loading, error, empty, hint }: { loading: boolean; error: string | null; empty?: boolean; hint?: string }) {
  if (loading) return <p className="sub">Loading…</p>;
  if (error) return <p className="note err">Could not reach the API: {error}. Start the backend with <code>uvicorn app.main:app --port 8000</code> from <code>backend/</code>.</p>;
  if (empty) return <p className="note">{hint ?? "No results yet."}</p>;
  return null;
}

export function Table({ cols, rows }: { cols: { key: string; label: string; num?: boolean; render?: (r: any) => ReactNode }[]; rows: any[] }) {
  return (
    <div className="scroll"><table>
      <thead><tr>{cols.map(c => <th key={c.key} className={c.num ? "num" : ""}>{c.label}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => <tr key={r.id ?? i}>{cols.map(c => <td key={c.key} className={c.num ? "num" : ""}>{c.render ? c.render(r) : r[c.key]}</td>)}</tr>)}</tbody>
    </table></div>
  );
}

export function ConfusionMatrix({ cm }: { cm: number[][] }) {
  return (<div className="cm" aria-label="confusion matrix: rows actual, columns predicted">
    <div>{cm[0][0]}<br /><small>true legit</small></div><div className="bad">{cm[0][1]}<br /><small>false alarm</small></div>
    <div className="bad">{cm[1][0]}<br /><small>missed fraud</small></div><div>{cm[1][1]}<br /><small>caught fraud</small></div>
  </div>);
}
