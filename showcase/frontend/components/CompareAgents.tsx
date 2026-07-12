"use client";

import { motion } from "framer-motion";
import { Bot, Shuffle, SplitSquareHorizontal } from "lucide-react";
import { useState } from "react";
import { EpisodeResult, runPolicy, videoSrc } from "@/lib/api";
import { Section } from "./ui/Section";
import { Reveal } from "./ui/Reveal";

type Mode = "random" | "best" | "both";

export function CompareAgents() {
  const [loading, setLoading] = useState<Mode | null>(null);
  const [random, setRandom] = useState<EpisodeResult | null>(null);
  const [best, setBest] = useState<EpisodeResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function compare(mode: Mode) {
    setLoading(mode);
    setError(null);
    try {
      if (mode === "random") setRandom(await runPolicy("random"));
      else if (mode === "best") setBest(await runPolicy("best"));
      else {
        // Run sequentially, not in parallel: the backend allows one inference at
        // a time (a second concurrent request returns 429, and two CarRacing
        // rollouts at once would OOM the free tier). Each result renders as it
        // arrives.
        setRandom(await runPolicy("random"));
        setBest(await runPolicy("best"));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Backend unavailable.");
    } finally {
      setLoading(null);
    }
  }

  const show = [
    random && { title: "Random policy", result: random, accent: false },
    best && { title: "Best PPO", result: best, accent: true },
  ].filter(Boolean) as { title: string; result: EpisodeResult; accent: boolean }[];

  return (
    <Section
      id="compare"
      eyebrow="04 · Compare Agents"
      title="Trained policy vs. random"
      lead="Run a random agent, the best PPO agent, or both side-by-side on the same track."
    >
      <Reveal>
        <div className="mx-auto max-w-4xl">
          <div className="flex flex-wrap justify-center gap-3">
            <Btn onClick={() => compare("random")} loading={loading === "random"} icon={Shuffle}>
              Random Agent
            </Btn>
            <Btn onClick={() => compare("best")} loading={loading === "best"} icon={Bot} primary>
              Best PPO
            </Btn>
            <Btn onClick={() => compare("both")} loading={loading === "both"} icon={SplitSquareHorizontal}>
              Both
            </Btn>
          </div>

          {error && (
            <p className="mt-4 text-center text-sm text-rose-600">{error}</p>
          )}

          {show.length > 0 && (
            <div className={`mt-8 grid gap-6 ${show.length === 2 ? "md:grid-cols-2" : ""}`}>
              {show.map((item) => (
                <motion.div
                  key={item.title}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4 }}
                  className="card overflow-hidden p-4"
                >
                  <div className="mb-3 flex items-center justify-between">
                    <span className="text-sm font-semibold text-slate-900">{item.title}</span>
                    <span
                      className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                        item.accent
                          ? "bg-brand-50 text-brand-700"
                          : "bg-slate-100 text-slate-500"
                      }`}
                    >
                      reward {item.result.reward.toFixed(1)}
                    </span>
                  </div>
                  <video
                    src={videoSrc(item.result)}
                    autoPlay
                    loop
                    muted
                    playsInline
                    className="w-full rounded-lg border border-slate-200 bg-slate-900"
                  />
                  <div className="mt-3 grid grid-cols-3 gap-2 text-center text-xs">
                    <Cell label="Reward" value={item.result.reward.toFixed(1)} />
                    <Cell label="Length" value={`${item.result.length}`} />
                    <Cell
                      label="Completion"
                      value={`${Math.round((item.result.length / 1000) * 100)}%`}
                    />
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </div>
      </Reveal>
    </Section>
  );
}

function Btn({
  children,
  onClick,
  loading,
  icon: Icon,
  primary,
}: {
  children: React.ReactNode;
  onClick: () => void;
  loading: boolean;
  icon: typeof Bot;
  primary?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={loading}
      className={`inline-flex items-center gap-2 rounded-xl px-5 py-3 text-sm font-medium transition disabled:opacity-60 ${
        primary
          ? "bg-brand-600 text-white hover:bg-brand-700"
          : "border border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
      }`}
    >
      <Icon className="h-4 w-4" />
      {loading ? "Running…" : children}
    </button>
  );
}

function Cell({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-slate-50 py-2">
      <div className="text-sm font-semibold text-slate-900">{value}</div>
      <div className="text-[10px] uppercase tracking-wide text-slate-400">{label}</div>
    </div>
  );
}
