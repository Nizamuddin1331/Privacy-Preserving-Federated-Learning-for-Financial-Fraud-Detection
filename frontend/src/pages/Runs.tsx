import { api } from "../services/api";
import { Badge, Page, Panel, State, Table, fmt, useApi } from "../components/ui";

export default function Runs() {
  const { data, error, loading, reload } = useApi(api.experiments, 2000);
  return (
    <Page title="Run queue" lead="Every experiment is stored with its settings and full round-by-round history. This list refreshes while jobs are running.">
      <Panel>
        <State loading={loading} error={error} empty={!!data && data.length === 0} hint="No runs yet. Use Launch runs to start one." />
        {data && data.length > 0 && <Table rows={data} cols={[
          { key: "id", label: "#" }, { key: "name", label: "Name" }, { key: "tag", label: "Study" }, { key: "mode", label: "Mode" },
          { key: "status", label: "Status", render: r => <Badge status={r.status} /> },
          { key: "progress", label: "Progress", num: true, render: r => `${Math.round(r.progress * 100)}%` },
          { key: "pr", label: "PR-AUC", num: true, render: r => fmt(r.final?.pr_auc) },
          { key: "f1", label: "F1", num: true, render: r => fmt(r.final?.f1) },
          { key: "eps", label: "ε", num: true, render: r => fmt(r.epsilon, 2) },
          { key: "err", label: "", render: r => r.error ? <span title={r.error} style={{ color: "var(--alert)" }}>error</span> : null },
          { key: "x", label: "", render: r => <button className="secondary" onClick={() => api.remove(r.id).then(reload)}>Delete</button> },
        ]} />}
      </Panel>
    </Page>
  );
}
