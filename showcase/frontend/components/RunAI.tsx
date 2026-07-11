"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Activity, AlertCircle, Clock, Gauge, PlayCircle, Ruler, Trophy } from "lucide-react";
import { useRef, useState } from "react";
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
  const [isSleeping, setIsSleeping] = useState(false);
  const [isDemo, setIsDemo] = useState(false);
  const runningRef = useRef(false); // Synchronous guard against double-clicks

  async function run() {
    if (runningRef.current) return; // Prevent double-click
    runningRef.current = true;
    setLoading(true);
    setError(null);
    setIsSleeping(false);
    setIsDemo(false);
    setResult(null);
    try {
      setResult(await runPolicy("best", 1028));
    } catch (e) {
      const msg =
        e instanceof Error ? e.message : "Could not reach the inference backend.";
      // Network errors (TypeError / failed to fetch) mean the backend is sleeping
      const sleeping =
        e instanceof TypeError || msg.toLowerCase().includes("fetch");
      setIsSleeping(sleeping);
      setError(msg);

      // Play cached demo automatically if backend fails
      setResult({
        policy: "demo",
        seed: 1028,
        reward: 598.4,
        length: 1000,
        inference_seconds: 0,
        steps_per_second: 0,
        // Since API_URL might be prepended in videoSrc, we'll handle this in the render logic or just use an absolute path if possible. 
        // Wait, videoSrc(result) prepends API_URL. We need a way to bypass that. Let's just set video_url to the demo path and we'll check isDemo in the render.
        video_url: "/demo.mp4",
      });
      setIsDemo(true);
    } finally {
      setLoading(false);
      runningRef.current = false;
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
                className={`mt-8 flex items-start gap-3 rounded-xl border p-4 text-sm ${
                  isSleeping
                    ? "border-amber-200 bg-amber-50 text-amber-700"
                    : "border-rose-200 bg-rose-50 text-rose-700"
                }`}
              >
                <AlertCircle className="mt-0.5 h-5 w-5 shrink-0" />
                <div>
                  {isSleeping ? (
                    <>
                      <p className="font-medium">Backend is waking up…</p>
                      <p className="mt-1 text-amber-600/80">
                        First request may take 20–30 seconds. Click &quot;Run
                        Evaluation&quot; again in a moment.
                      </p>
                    </>
                  ) : (
                    <>
                      <p className="font-medium">Inference backend unavailable.</p>
                      <p className="mt-1 text-rose-600/80">{error}</p>
                      <p className="mt-2 text-xs text-rose-500/80">
                        Start it with <code>uvicorn main:app</code> in <code>showcase/backend</code>.
                      </p>
                    </>
                  )}
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
                <div className="relative lg:col-span-3">
                  <video
                    key={result.video_url}
                    src={isDemo ? result.video_url : videoSrc(result)}
                    autoPlay
                    loop
                    muted
                    playsInline
                    controls
                    className="w-full rounded-xl border border-slate-200 bg-slate-900"
                  />
                  {isDemo && (
                    <div className="absolute right-3 top-3 rounded-full bg-black/60 px-2.5 py-1 text-xs font-medium text-white backdrop-blur-sm">
                      Cached Demo
                    </div>
                  )}
                </div>
                <div className="grid grid-cols-2 gap-3 lg:col-span-2 lg:grid-cols-1">
                  <Metric icon={Trophy} label="Reward" value={result.reward} decimals={1} accent />
                  <Metric icon={Ruler} label="Episode length" value={result.length} unit="steps" />
                  {isDemo ? (
                    <>
                      <Metric icon={Clock} label="Inference time" value={0} overrideValue="--" />
                      <Metric icon={Gauge} label="Throughput" value={0} overrideValue="--" />
                    </>
                  ) : (
                    <>
                      <Metric icon={Clock} label="Inference time" value={result.inference_seconds} decimals={2} unit="s" />
                      <Metric icon={Gauge} label="Throughput" value={result.steps_per_second} decimals={1} unit="steps/s" />
                    </>
                  )}
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
  overrideValue,
  unit,
  decimals = 0,
  accent,
}: {
  icon: typeof Trophy;
  label: string;
  value: number;
  overrideValue?: string;
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
        {overrideValue ? (
          overrideValue
        ) : (
          <AnimatedNumber value={value} decimals={decimals} />
        )}
        {unit && <span className="ml-1 text-sm font-normal text-slate-400">{unit}</span>}
      </div>
    </div>
  );
}
