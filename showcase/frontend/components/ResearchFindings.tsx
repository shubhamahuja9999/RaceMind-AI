"use client";

import { Lightbulb, LineChart, XCircle } from "lucide-react";
import { FINDINGS } from "@/lib/data";
import { Section } from "./ui/Section";
import { Reveal } from "./ui/Reveal";

export function ResearchFindings() {
  return (
    <Section
      id="findings"
      eyebrow="06 · Research Findings"
      title="Honest results — including the negatives"
      lead="Every controlled change tested so far regressed against the baseline. Here is what happened, and why."
    >
      <div className="grid gap-6 md:grid-cols-3">
        {FINDINGS.map((f, i) => (
          <Reveal key={f.finding} delay={i * 0.08}>
            <div className="card flex h-full flex-col p-6">
              <div className="flex items-center gap-2 text-rose-600">
                <XCircle className="h-5 w-5" />
                <span className="text-sm font-semibold">{f.finding}</span>
              </div>

              <div className="mt-5 space-y-4 text-sm">
                <div>
                  <div className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-slate-400">
                    <LineChart className="h-3.5 w-3.5" /> Evidence
                  </div>
                  <p className="mt-1 font-mono text-[13px] text-slate-700">{f.evidence}</p>
                </div>
                <div>
                  <div className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-slate-400">
                    <Lightbulb className="h-3.5 w-3.5" /> Interpretation
                  </div>
                  <p className="mt-1 leading-relaxed text-slate-600">{f.interpretation}</p>
                </div>
              </div>
            </div>
          </Reveal>
        ))}
      </div>

      <Reveal delay={0.15}>
        <p className="mx-auto mt-8 max-w-2xl text-center text-sm text-slate-500">
          No easy win was found. The contribution is the <span className="font-medium text-slate-700">method</span> —
          reproducible, statistically tested, honestly reported negatives — not a claim of a better agent.
        </p>
      </Reveal>
    </Section>
  );
}
