"use client";

import { AnimatePresence, motion } from "framer-motion";
import { ChevronRight } from "lucide-react";
import { useState } from "react";
import { EXPERIMENTS } from "@/lib/data";
import { Section } from "./ui/Section";
import { Reveal } from "./ui/Reveal";
import { VerdictBadge } from "./ui/VerdictBadge";

export function ExperimentTimeline() {
  const [open, setOpen] = useState<string | null>("baseline");

  return (
    <Section
      id="timeline"
      eyebrow="02 · Experiment Timeline"
      title="Every experiment, one variable at a time"
      lead="Each study changes exactly one thing versus the frozen baseline. Click a card to expand the details."
    >
      <div className="relative mx-auto max-w-3xl">
        {/* vertical connector */}
        <div className="absolute left-[19px] top-2 bottom-2 w-px bg-slate-200 sm:left-[23px]" />
        <div className="space-y-4">
          {EXPERIMENTS.map((exp, i) => {
            const isOpen = open === exp.id;
            return (
              <Reveal key={exp.id} delay={i * 0.05}>
                <div className="relative pl-12 sm:pl-14">
                  <span className="absolute left-0 top-4 flex h-10 w-10 items-center justify-center rounded-full border border-slate-200 bg-white text-sm font-semibold text-slate-500 shadow-sm sm:h-12 sm:w-12">
                    {i + 1}
                  </span>
                  <button
                    onClick={() => setOpen(isOpen ? null : exp.id)}
                    className="card w-full p-5 text-left"
                    aria-expanded={isOpen}
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <div className="flex flex-wrap items-center gap-2">
                          <h3 className="text-base font-semibold text-slate-900">{exp.name}</h3>
                          <VerdictBadge verdict={exp.verdict} />
                        </div>
                        <p className="mt-1 text-xs font-medium uppercase tracking-wide text-slate-400">
                          {exp.variable}
                        </p>
                      </div>
                      <ChevronRight
                        className={`mt-1 h-5 w-5 shrink-0 text-slate-400 transition-transform ${
                          isOpen ? "rotate-90" : ""
                        }`}
                      />
                    </div>

                    <div className="mt-3 grid gap-3 sm:grid-cols-2">
                      <Field label="Hypothesis" value={exp.hypothesis} />
                      <Field label="Result" value={exp.result} accent />
                    </div>

                    <AnimatePresence initial={false}>
                      {isOpen && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: "auto", opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ duration: 0.3, ease: "easeInOut" }}
                          className="overflow-hidden"
                        >
                          <div className="mt-4 rounded-xl bg-slate-50 p-4">
                            <p className="text-sm leading-relaxed text-slate-600">{exp.details}</p>
                            <p className="mt-3 text-xs font-medium text-slate-400">
                              Confidence · {exp.confidence}
                            </p>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </button>
                </div>
              </Reveal>
            );
          })}
        </div>
      </div>
    </Section>
  );
}

function Field({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div className="text-xs font-medium uppercase tracking-wide text-slate-400">{label}</div>
      <div className={`mt-0.5 text-sm ${accent ? "font-semibold text-slate-900" : "text-slate-600"}`}>
        {value}
      </div>
    </div>
  );
}
