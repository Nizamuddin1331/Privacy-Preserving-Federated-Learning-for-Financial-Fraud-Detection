import { api } from "../services/api";
import { Page, Panel, State, Table, fmt, useApi } from "../components/ui";

export default function SecureAggregation() {
  const { data, error, loading } = useApi(api.secagg);
  return (
    <Page title="Secure aggregation" lead="Differential privacy protects against what the final model reveals. Secure aggregation protects each bank's update on its way to the server.">
      <Panel title="How it works">
        <ol style={{ margin: 0, paddingLeft: 20, maxWidth: "75ch" }}>
          <li>Every pair of banks agrees a secret using an X25519 key exchange.</li>
          <li>Each bank adds the pairwise masks to its update before uploading; for each pair one bank adds the mask and the other subtracts it.</li>
          <li>The server adds up the uploads. The masks cancel, leaving only the sum of the updates.</li>
          <li>If a bank drops out, the survivors reveal their shared secrets with that bank so the server can remove the leftover masks.</li>
        </ol>
        <p className="note">This is a simplified version of the Bonawitz et al. protocol: no secret-sharing of seeds and an honest-but-curious server.</p>
      </Panel>
      <State loading={loading} error={error} empty={!!data && data.length === 0} hint="Run modes C, D and E to compare the cost." />
      {data && data.length > 0 && <Panel title="What it costs" sub="Accuracy should match the same run without secure aggregation. The price is bandwidth (64-bit masked values instead of 32-bit) and compute.">
        <Table rows={data} cols={[{ key: "name", label: "Run" }, { key: "mode", label: "Mode" }, { key: "secure", label: "SecAgg", render: r => (r.secure ? "yes" : "no") }, { key: "dp", label: "DP", render: r => (r.dp ? "yes" : "no") },
          { key: "dropout_prob", label: "Dropout", num: true }, { key: "dropped_total", label: "Dropped uploads", num: true }, { key: "pr_auc", label: "PR-AUC", num: true, render: r => fmt(r.pr_auc) },
          { key: "uplink_mb", label: "Uplink MB", num: true, render: r => fmt(r.uplink_mb, 2) }, { key: "runtime_s", label: "Seconds", num: true, render: r => fmt(r.runtime_s, 1) }]} />
      </Panel>}
    </Page>
  );
}
