"use client";

import { motion } from "framer-motion";
import { ArrowRight, Github, PlayCircle, Sparkles, Tag } from "lucide-react";
import { LINKS } from "@/lib/data";

// A few subtle "telemetry" dots drifting behind the hero (very minimal motion).
const DOTS = [
  { x: "12%", y: "28%", d: 0 },
  { x: "82%", y: "20%", d: 1.2 },
  { x: "68%", y: "70%", d: 0.6 },
  { x: "24%", y: "72%", d: 1.8 },
  { x: "48%", y: "18%", d: 0.9 },
];

export function Hero() {
  return (
    <header className="relative overflow-hidden border-b border-slate-100">
      <div className="grid-bg absolute inset-0 -z-10" />
      <div className="absolute inset-0 -z-10 bg-gradient-to-b from-white/0 via-white/40 to-white" />

      {DOTS.map((dot, i) => (
        <motion.span
          key={i}
          className="absolute h-2 w-2 rounded-full bg-telemetry-400/60"
          style={{ left: dot.x, top: dot.y }}
          animate={{ y: [0, -14, 0], opacity: [0.4, 0.9, 0.4] }}
          transition={{ duration: 5 + dot.d, repeat: Infinity, ease: "easeInOut" }}
        />
      ))}

      <div className="section pb-20 pt-28 sm:pt-36">
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          className="mx-auto max-w-3xl text-center"
        >
          <span className="eyebrow">
            <Sparkles className="h-3.5 w-3.5 text-brand-500" />
            RaceMind AI Research Lab
          </span>

          <h1 className="mt-6 text-5xl font-semibold tracking-tight text-slate-900 sm:text-6xl">
            RaceMind AI
          </h1>
          <p className="mx-auto mt-5 max-w-2xl text-lg leading-relaxed text-slate-600 sm:text-xl">
            A Modular Reinforcement Learning Research Platform for Autonomous Racing.
          </p>

          <div className="mt-9 flex flex-wrap items-center justify-center gap-3">
            <a
              href="#run"
              className="inline-flex items-center gap-2 rounded-xl bg-brand-600 px-5 py-3 text-sm font-medium text-white shadow-sm transition hover:bg-brand-700"
            >
              <PlayCircle className="h-4 w-4" /> Run AI
            </a>
            <a
              href="#timeline"
              className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-5 py-3 text-sm font-medium text-slate-700 transition hover:border-slate-300 hover:bg-slate-50"
            >
              Explore Research <ArrowRight className="h-4 w-4" />
            </a>
            <a
              href={LINKS.github}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-5 py-3 text-sm font-medium text-slate-700 transition hover:border-slate-300 hover:bg-slate-50"
            >
              <Github className="h-4 w-4" /> GitHub
            </a>
            <a
              href={LINKS.release}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm font-medium text-slate-500 transition hover:border-slate-300 hover:bg-slate-50"
            >
              <Tag className="h-4 w-4" /> Release v1.0
            </a>
          </div>

          <div className="mx-auto mt-12 flex max-w-xl items-center justify-center gap-8 text-sm text-slate-500">
            <Stat value="598.42" label="baseline reward" />
            <div className="h-8 w-px bg-slate-200" />
            <Stat value="4" label="controlled experiments" />
            <div className="h-8 w-px bg-slate-200" />
            <Stat value="1M" label="training steps" />
          </div>
        </motion.div>
      </div>
    </header>
  );
}

function Stat({ value, label }: { value: string; label: string }) {
  return (
    <div className="text-center">
      <div className="text-2xl font-semibold text-slate-900">{value}</div>
      <div className="mt-0.5 text-xs uppercase tracking-wide text-slate-400">{label}</div>
    </div>
  );
}
