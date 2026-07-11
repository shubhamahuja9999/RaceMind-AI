"use client";

import { useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import trainingData from "@/lib/data/training.json";
import { Section } from "./ui/Section";
import { Reveal } from "./ui/Reveal";

type MetricKey = "trainReward" | "evalReward" | "loss" | "fps" | "lr";

const METRICS: {
  key: MetricKey;
  label: string;
  source: "training" | "eval";
  dataKey: string;
  color: string;
  decimals: number;
}[] = [
  { key: "trainReward", label: "Training Reward", source: "training", dataKey: "trainReward", color: "#2563eb", decimals: 0 },
  { key: "evalReward", label: "Evaluation Reward", source: "eval", dataKey: "reward", color: "#3b82f6", decimals: 0 },
  { key: "loss", label: "Loss", source: "training", dataKey: "loss", color: "#64748b", decimals: 3 },
  { key: "fps", label: "FPS", source: "training", dataKey: "fps", color: "#f97316", decimals: 0 },
  { key: "lr", label: "Learning Rate", source: "training", dataKey: "lr", color: "#0ea5e9", decimals: 5 },
];

const fmtStep = (s: number) =>
  s >= 1_000_000 ? `${(s / 1e6).toFixed(1)}M` : `${Math.round(s / 1000)}k`;

export function TrainingDashboard() {
  const [active, setActive] = useState<MetricKey>("trainReward");
  const metric = METRICS.find((m) => m.key === active)!;
  const data = metric.source === "eval" ? trainingData.evalReward : trainingData.training;

  return (
    <Section
      id="dashboard"
      eyebrow="05 · Training Dashboard"
      title="Training dynamics, from real logs"
      lead="Interactive curves recorded during training — reward, loss, throughput and the learning rate."
    >
      <Reveal>
        <div className="card mx-auto max-w-4xl p-6 sm:p-8">
          <div className="mb-6 flex flex-wrap gap-2">
            {METRICS.map((m) => (
              <button
                key={m.key}
                onClick={() => setActive(m.key)}
                className={`rounded-lg px-3.5 py-1.5 text-sm font-medium transition ${
                  active === m.key
                    ? "bg-slate-900 text-white"
                    : "border border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data} margin={{ top: 8, right: 8, left: -8, bottom: 0 }}>
                <defs>
                  <linearGradient id="fill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={metric.color} stopOpacity={0.25} />
                    <stop offset="100%" stopColor={metric.color} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#eef2f6" vertical={false} />
                <XAxis
                  dataKey="step"
                  tickFormatter={fmtStep}
                  stroke="#94a3b8"
                  fontSize={12}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  stroke="#94a3b8"
                  fontSize={12}
                  tickLine={false}
                  axisLine={false}
                  width={56}
                  tickFormatter={(v: number) =>
                    metric.key === "lr" ? v.toExponential(0) : `${v}`
                  }
                />
                <Tooltip
                  contentStyle={{
                    borderRadius: 12,
                    border: "1px solid #e2e8f0",
                    boxShadow: "0 8px 24px -8px rgb(15 23 42 / 0.12)",
                    fontSize: 13,
                  }}
                  labelFormatter={(v: number) => `Step ${fmtStep(v)}`}
                  formatter={(v: number) => [Number(v).toFixed(metric.decimals), metric.label]}
                />
                <Area
                  type="monotone"
                  dataKey={metric.dataKey}
                  stroke={metric.color}
                  strokeWidth={2}
                  fill="url(#fill)"
                  animationDuration={600}
                  dot={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-4 text-center text-xs text-slate-400">
            Training metrics from a 1M-step PPO run; evaluation reward from checkpoint
            evaluations of the baseline scale-up.
          </p>
        </div>
      </Reveal>
    </Section>
  );
}
