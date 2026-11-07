// Thin fetch client. /api is proxied to FastAPI in dev (see vite.config.ts).
export interface Final { precision: number; recall: number; f1: number; pr_auc: number; roc_auc: number; fpr: number; fnr: number }
export interface Experiment {
  id: number; name: string; mode: string; tag: string; status: "queued" | "running" | "completed" | "failed";
  progress: number; config: Record<string, any>; error?: string | null; created_at: string;
  final?: Final | null; epsilon?: number | null;
}
export interface CompRow extends Final { id: number; mode: string; label: string; epsilon: number | null; uplink_mb: number; runtime_s: number; confusion_matrix: number[][] }
export interface Series { id: number; mode: string; label: string; name: string; history: { round: number; pr_auc: number; f1: number; recall: number; precision: number; train_loss: number; epsilon: number | null }[] }
export interface PrivacyRow { id: number; name: string; tag: string; noise_multiplier: number; epsilon: number; delta: number; pr_auc: number; f1: number; recall: number; max_grad_norm: number }
export interface SecRow { id: number; name: string; tag: string; mode: string; secure: boolean; dp: boolean; dropout_prob: number; pr_auc: number; uplink_mb: number; runtime_s: number; dropped_total: number }
export interface RobRow { id: number; attack: string; aggregator: string; secure: boolean; n_malicious: number; pr_auc: number; f1: number; recall: number }
export interface DatasetInfo { source: string; n_rows: number; n_features: number; n_fraud: number; n_legit: number; fraud_rate: number; amount_mean: number; amount_median: number; amount_max: number; csv_present: boolean; expected_path: string; head: Record<string, number>[] }
export interface PartitionRow { client: number; n: number; n_fraud: number; fraud_rate: number }

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`/api${path}`, { headers: { "Content-Type": "application/json" }, ...init });
  if (!r.ok) {
    let detail = r.statusText;
    try { const j = await r.json(); detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch { /* keep statusText */ }
    throw new Error(`${r.status}: ${detail}`);
  }
  return r.status === 204 ? (undefined as T) : r.json();
}
const post = <T,>(p: string, body: unknown) => req<T>(p, { method: "POST", body: JSON.stringify(body) });

export const api = {
  health: () => req<{ status: string; torch: string; database: string; experiments: Record<string, number> }>("/health"),
  dataset: () => req<DatasetInfo>("/dataset/info"),
  partitions: (n: number, strategy: string, alpha: number) => req<{ clients: PartitionRow[] }>(`/dataset/partition-preview?n_clients=${n}&strategy=${strategy}&alpha=${alpha}`),
  experiments: () => req<Experiment[]>("/experiments"),
  launch: (b: object) => post<{ ids: number[] }>("/experiments", b),
  launchSuite: (b: object) => post<{ ids: number[] }>("/experiments/suite", b),
  launchSweep: (base: object, noise_multipliers: number[]) => post<{ ids: number[] }>("/experiments/sweep", { base, noise_multipliers }),
  launchRobustness: (base: object, n_malicious: number) => post<{ ids: number[] }>("/experiments/robustness", { base, n_malicious }),
  remove: (id: number) => req<void>(`/experiments/${id}`, { method: "DELETE" }),
  comparison: () => req<CompRow[]>("/metrics/comparison"),
  convergence: () => req<Series[]>("/metrics/convergence"),
  privacy: () => req<PrivacyRow[]>("/metrics/privacy"),
  secagg: () => req<SecRow[]>("/metrics/secure-aggregation"),
  robustness: () => req<RobRow[]>("/metrics/robustness"),
};
