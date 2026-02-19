import { useParams } from "@solidjs/router";
import KataWorkspace from "../components/kata-workspace/kata-workspace";
import { PHASE_NAMES } from "../lib/constants";
import "./kata-page.css";

export default function KataPage() {
  const params = useParams();

  const phaseNum = () => parseInt(params.phaseId, 10);
  const phaseName = () => PHASE_NAMES[phaseNum()] ?? "";
  const kataId = () => params.kataId?.replace(/-/g, " ") ?? "";

  return (
    <div class="kata-page">
      <div class="kata-page__header">
        <span class="kata-page__phase">Phase {params.phaseId} — {phaseName()}</span>
        <h1 class="kata-page__title">{kataId()}</h1>
      </div>
      <KataWorkspace />
    </div>
  );
}
