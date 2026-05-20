import Nav from "./components/Nav";
import Hero from "./sections/Hero";
import Problem from "./sections/Problem";
import Pipeline from "./sections/Pipeline";
import Protocol from "./sections/Protocol";
import Results from "./sections/Results";
import Takeaway from "./sections/Takeaway";
import Repro from "./sections/Repro";
import snapshot from "./data/snapshot.json";
import type { SiteSnapshot } from "./types";

const data = snapshot as unknown as SiteSnapshot;

const sections = [
  { id: "hero", label: "Overview" },
  { id: "problem", label: "Problem" },
  { id: "pipeline", label: "Pipeline" },
  { id: "protocol", label: "Protocol" },
  { id: "results", label: "Results" },
  { id: "takeaway", label: "Takeaway" },
  { id: "repro", label: "Repro" },
];

export default function App() {
  return (
    <div className="layout">
      <Nav sections={sections} generatedAt={data.generated_at} />
      <main>
        <Hero scorecard={data.scorecard} />
        <Problem />
        <Pipeline />
        <Protocol />
        <Results data={data} />
        <Takeaway />
        <Repro />
        <div className="foot">
          Snapshot generated {data.generated_at}. Numbers come from{" "}
          <code>data/sft/libero_4suite_v3</code> and{" "}
          <code>data/eval/steerability_v1_vs_v3</code>.
        </div>
      </main>
    </div>
  );
}
