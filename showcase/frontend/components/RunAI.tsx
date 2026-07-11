"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Activity, AlertCircle, Clock, Gauge, PlayCircle, Ruler, Trophy } from "lucide-react";
import { useState } from "react";
import { EpisodeResult, runPolicy, videoSrc } from "@/lib/api";
import { Section } from "./ui/Section";
import { Reveal } from "./ui/Reveal";
import { AnimatedNumber } from "./ui/AnimatedNumber";

const STEPS = [
  "Loading the evaluation environment",
  "Running one deterministic episode",
  "Collecting rendered frames",
  "Encoding the MP4",
];

export function RunAI() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EpisodeResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await runPolicy("best", 1028));
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Could not reach the inference backend."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <Section
      id="run"
      eyebrow="03 · Run the AI"
      title="Watch the trained agent drive — live"
      lead="One click runs a single deterministic episode of the trained PPO agent on the backend and streams back the video and telemetry."
    >
      <Reveal>
        <div className="card mx-auto max-w-4xl overflow-hidden p-6 sm:p-8">
          <div className="flex flex-col items-center gap-4 text-center">
            <button
              onClick={run}
              disabled={loading}
              className="inline-flex items-center gap-2 rounded-xl bg-brand-600 px-6 py-3 text-sm font-medium text-white shadow-sm transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-60"
            >
              <PlayCircle className="h-5 w-5" />
              {loading ? "Running…" : result ? "Run again" : "Run Evaluation"}
            </button>
            <p className="text-xs text-slate-400">
              PPO · CarRacing-v3 · deterministic · first run may take ~10–20s (CPU),
              then it is cached.
            </p>
          </div>

          <AnimatePresence mode="wait">
            {loading && <LoadingState key="loading" />}

            {error && !loading && (
              <motion.div
                key="error"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mt-8 flex items-start gap-3 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700"
              >
                <AlertCircle className="mt-0.5 h-5 w-5 shrink-0" />
                <div>
                  <p className="font-medium">Inference backend unavailable.</p>
                  <p className="mt-1 text-rose-600/80">{error}</p>
                  <p className="mt-2 text-xs text-rose-500/80">
                    Start it with <code>uvicorn main:app</code> in <code>showcase/backend</code>.
                  </p>
                </div>
              </motion.div>
            )}

            {result && !loading && (
              <motion.div
                key="result"
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
                className="mt-8 grid gap-6 lg:grid-cols-5"
              >
                <div className="lg:col-span-3">
                  <video
                    key={result.video_url}
                    src={videoSrc(result)}
                    autoPlay
                    loop
                    muted
                    playsInline
                    controls
                    className="w-full rounded-xl border border-slate-200 bg-slate-900"
                  />
                </div>
                <div className="grid grid-cols-2 gap-3 lg:col-span-2 lg:grid-cols-1">
                  <Metric icon={Trophy} label="Reward" value={result.reward} decimals={1} accent />
                  <Metric icon={Ruler} label="Episode length" value={result.length} unit="steps" />
                  <Metric icon={Clock} label="Inference time" value={result.inference_seconds} decimals={2} unit="s" />
                  <Metric icon={Gauge} label="Throughput" value={result.steps_per_second} decimals={1} unit="steps/s" />
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </Reveal>
    </Section>
  );
}

function LoadingState() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="mt-8"
    >
      <div className="mx-auto flex max-w-md flex-col gap-3">
        {STEPS.map((step, i) => (
          <motion.div
            key={step}
            initial={{ opacity: 0.3 }}
            animate={{ opacity: [0.3, 1, 0.3] }}
            transition={{ duration: 1.6, repeat: Infinity, delay: i * 0.4 }}
            className="flex items-center gap-3 text-sm text-slate-500"
          >
            <Activity className="h-4 w-4 text-telemetry-500" />
            {step}
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

function Metric({
  icon: Icon,
  label,
  value,
  unit,
  decimals = 0,
  accent,
}: {
  icon: typeof Trophy;
  label: string;
  value: number;
  unit?: string;
  decimals?: number;
  accent?: boolean;
}) {
  return (
    <div className={`rounded-xl border p-4 ${accent ? "border-telemetry-400/40 bg-telemetry-500/[0.04]" : "border-slate-200 bg-white"}`}>
      <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-slate-400">
        <Icon className={`h-4 w-4 ${accent ? "text-telemetry-500" : "text-slate-400"}`} />
        {label}
      </div>
      <div className="mt-1.5 text-2xl font-semibold text-slate-900">
        <AnimatedNumber value={value} decimals={decimals} />
        {unit && <span className="ml-1 text-sm font-normal text-slate-400">{unit}</span>}
      </div>
    </div>
  );
}
