"use client";

import { TECH } from "@/lib/data";
import { Section } from "./ui/Section";
import { Reveal } from "./ui/Reveal";

export function TechStack() {
  return (
    <Section
      id="tech"
      eyebrow="07 · Tech Stack"
      title="Built with production ML tooling"
    >
      <div className="mx-auto grid max-w-4xl grid-cols-2 gap-3 sm:grid-cols-3">
        {TECH.map((t, i) => (
          <Reveal key={t.name} delay={i * 0.04}>
            <div className="card flex items-center gap-3 p-4">
              <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-100 font-mono text-sm font-semibold text-slate-500">
                {t.name[0]}
              </span>
              <div>
                <div className="text-sm font-medium text-slate-900">{t.name}</div>
                <div className="text-xs text-slate-400">{t.note}</div>
              </div>
            </div>
          </Reveal>
        ))}
      </div>
    </Section>
  );
}
