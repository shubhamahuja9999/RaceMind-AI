"use client";

import { motion } from "framer-motion";
import { ChevronDown, FlaskConical, Layers, Target } from "lucide-react";
import { ARCHITECTURE } from "@/lib/data";
import { Section } from "./ui/Section";
import { Reveal } from "./ui/Reveal";

const PILLARS = [
  {
    icon: Target,
    title: "The problem",
    body: "PPO is high-variance. On a limited CPU budget, single-run comparisons routinely mislead — 'this helps' is usually a hunch, not a measurement.",
  },
  {
    icon: FlaskConical,
    title: "The approach",
    body: "Treat every run as a controlled experiment: change one variable, hold everything else to a frozen baseline, and judge the result with a paired statistical test.",
  },
  {
    icon: Layers,
    title: "The architecture",
    body: "A layered, interface-driven platform. Agents and rewards are pluggable; evaluation is decoupled from training, so results stay comparable.",
  },
];

export function Overview() {
  return (
    <Section
      id="overview"
      eyebrow="01 · Project Overview"
      title="Reinforcement learning you can actually falsify"
      lead="RaceMind AI is a research platform, not a training script — built to prove whether a change truly improves an agent."
    >
      <div className="grid gap-6 md:grid-cols-3">
        {PILLARS.map((p, i) => (
          <Reveal key={p.title} delay={i * 0.08}>
            <div className="card h-full p-6">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand-50 text-brand-600">
                <p.icon className="h-5 w-5" />
              </div>
              <h3 className="mt-4 text-lg font-semibold text-slate-900">{p.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-600">{p.body}</p>
            </div>
          </Reveal>
        ))}
      </div>

      {/* Architecture pipeline — clean SVG-style blocks. */}
      <Reveal delay={0.1}>
        <div className="card mt-10 p-6 sm:p-10">
          <p className="mb-8 text-center text-sm font-medium uppercase tracking-wide text-slate-400">
            System pipeline
          </p>
          <div className="mx-auto flex max-w-md flex-col items-center gap-1">
            {ARCHITECTURE.map((block, i) => (
              <div key={block.label} className="flex w-full flex-col items-center">
                <motion.div
                  initial={{ opacity: 0, scale: 0.96 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.35, delay: i * 0.06 }}
                  className="w-full rounded-xl border border-slate-200 bg-gradient-to-b from-white to-slate-50 px-5 py-3.5 text-center shadow-sm"
                >
                  <div className="font-medium text-slate-900">{block.label}</div>
                  <div className="text-xs text-slate-500">{block.note}</div>
                </motion.div>
                {i < ARCHITECTURE.length - 1 && (
                  <ChevronDown className="my-1 h-4 w-4 text-slate-300" />
                )}
              </div>
            ))}
          </div>
        </div>
      </Reveal>
    </Section>
  );
}
