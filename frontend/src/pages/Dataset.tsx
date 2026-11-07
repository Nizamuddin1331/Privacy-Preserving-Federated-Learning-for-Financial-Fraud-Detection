import { api } from "../services/api";
import { Page, Panel, Stat, State, Table, useApi } from "../components/ui";

export default function Dataset() {
  const { data, error, loading } = useApi(api.dataset);
  return (
    <Page title="Transactions" lead="The credit-card fraud benchmark has 284,807 transactions of which 492 are fraud. Without that file, a synthetic set with the same columns is used.">
      <State loading={loading} error={error} />
      {data && (<>
        {!data.csv_present && <p className="note">creditcard.csv was not found, so synthetic data is in use. To use the real dataset, place it at <code>{data.expected_path}</code>.</p>}
        <div className="stats">
          <Stat label="Source" value={data.source} /><Stat label="Transactions" value={data.n_rows.toLocaleString()} />
          <Stat label="Fraud cases" value={data.n_fraud.toLocaleString()} /><Stat label="Fraud rate" value={`${(data.fraud_rate * 100).toFixed(2)}%`} />
          <Stat label="Median amount" value={data.amount_median.toFixed(2)} /><Stat label="Largest amount" value={data.amount_max.toFixed(0)} />
        </div>
        <Panel title="First rows" sub="V1–V28 are anonymised PCA components, so no bank can recognise a customer from them.">
          <Table cols={["Time", "V1", "V2", "V3", "V4", "Amount", "Class"].map(k => ({ key: k, label: k, num: true }))} rows={data.head} />
        </Panel>
      </>)}
    </Page>
  );
}
