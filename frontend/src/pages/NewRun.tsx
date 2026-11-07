import RunForm from "../components/RunForm";
import { Page, Panel } from "../components/ui";

export default function NewRun() {
  return (
    <Page title="Launch runs" lead="Pick the federation settings, then run one mode or a whole study. Jobs run one at a time in the background.">
      <Panel title="Settings" sub="Run all modes A–E for the headline comparison. The privacy sweep varies the noise level, and the poisoning grid tests how the server copes with a dishonest bank.">
        <RunForm onLaunched={() => { location.hash = "#/runs"; }} />
      </Panel>
    </Page>
  );
}
