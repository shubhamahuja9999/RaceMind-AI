import { Navbar } from "@/components/Navbar";
import { Hero } from "@/components/Hero";
import { Overview } from "@/components/Overview";
import { ExperimentTimeline } from "@/components/ExperimentTimeline";
import { RunAI } from "@/components/RunAI";
import { CompareAgents } from "@/components/CompareAgents";
import { TrainingDashboard } from "@/components/TrainingDashboard";
import { ResearchFindings } from "@/components/ResearchFindings";
import { TechStack } from "@/components/TechStack";
import { Repository } from "@/components/Repository";
import { Footer } from "@/components/Footer";

export default function Home() {
  return (
    <main id="top">
      <Navbar />
      <Hero />
      <Overview />
      <ExperimentTimeline />
      <RunAI />
      <CompareAgents />
      <TrainingDashboard />
      <ResearchFindings />
      <TechStack />
      <Repository />
      <Footer />
    </main>
  );
}
