import { useState } from "react";
import { api } from "../services/api";

export interface RunSettings {
  n_synthetic: number; n_clients: number; partition: string; dirichlet_alpha: number; rounds: number;
  local_epochs: number; noise_multiplier: number; dropout_prob: number; mode: string;
}
export const DEFAULTS: RunSettings = { n_synthetic: 20000, n_clients: 5, partition: "iid", dirichlet_alpha: 0.5, rounds: 10, local_epochs: 2, noise_multiplier: 1.1, dropout_prob: 0, mode: "E" };

export default function RunForm({ onLaunched }: { onLaunched: () => void }) {
  const [s, setS] = useState<RunSettings>(DEFAULTS);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const num = (k: keyof RunSettings) => (e: React.ChangeEvent<HTMLInputElement>) => setS({ ...s, [k]: Number(e.target.value) });
  const txt = (k: keyof RunSettings) => (e: React.ChangeEvent<HTMLSelectElement>) => setS({ ...s, [k]: e.target.value });

  const go = async (fn: () => Promise<unknown>) => {
    setBusy(true); setErr(null);
    try { await fn(); onLaunched(); } catch (e) { setErr(String((e as Error).message)); } finally { setBusy(false); }
  };
  return (
    <>
      <div className="form">
        <label>Training mode<select value={s.mode} onChange={txt("mode")}>
          <option value="A">A – Centralised</option><option value="B">B – Local-only</option><option value="C">C – FedAvg</option>
          <option value="D">D – FedAvg + DP</option><option value="E">E – FedAvg + DP + SecAgg</option></select></label>
        <label>Banks<input type="number" min={2} max={20} value={s.n_clients} onChange={num("n_clients")} /></label>
        <label>Data split<select value={s.partition} onChange={txt("partition")}>
          <option value="iid">IID</option><option value="dirichlet">Non-IID (Dirichlet)</option><option value="prevalence">Non-IID (fraud prevalence)</option></select></label>
        <label>Dirichlet alpha (lower = more skew)<input type="number" step={0.1} min={0.1} value={s.dirichlet_alpha} onChange={num("dirichlet_alpha")} /></label>
        <label>Rounds<input type="number" min={1} max={100} value={s.rounds} onChange={num("rounds")} /></label>
        <label>Local epochs<input type="number" min={1} max={10} value={s.local_epochs} onChange={num("local_epochs")} /></label>
        <label>DP noise multiplier (σ)<input type="number" step={0.1} min={0.2} value={s.noise_multiplier} onChange={num("noise_multiplier")} /></label>
        <label>Bank dropout per round<input type="number" step={0.1} min={0} max={0.9} value={s.dropout_prob} onChange={num("dropout_prob")} /></label>
        <label>Synthetic transactions<input type="number" step={5000} min={2000} value={s.n_synthetic} onChange={num("n_synthetic")} /></label>
      </div>
      {err && <p className="note err">{err}</p>}
      <div className="actions">
        <button disabled={busy} onClick={() => go(() => api.launch(s))}>Run mode {s.mode}</button>
        <button disabled={busy} className="secondary" onClick={() => go(() => api.launchSuite(s))}>Run all modes A–E</button>
        <button disabled={busy} className="secondary" onClick={() => go(() => api.launchSweep({ ...s, mode: "D" }, [0.6, 0.9, 1.1, 1.6, 2.5]))}>Run privacy sweep</button>
        <button disabled={busy} className="secondary" onClick={() => go(() => api.launchRobustness({ ...s, mode: "C" }, 1))}>Run poisoning grid</button>
      </div>
    </>
  );
}
