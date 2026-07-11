// Static, real data for the RaceMind AI Research Lab showcase.
// Numbers come from the project's archived experiment reports.

export const LINKS = {
  github: "https://github.com/shubhamahuja9999/RaceMind-AI",
  release: "https://github.com/shubhamahuja9999/RaceMind-AI/releases/tag/v1.0.0",
  docs: "https://github.com/shubhamahuja9999/RaceMind-AI/tree/main/docs",
  report:
    "https://github.com/shubhamahuja9999/RaceMind-AI/blob/main/research/final_report.md",
};

export const BASELINE = {
  mean: 598.42,
  ci: [544.78, 652.05] as [number, number],
};

export type Verdict = "Baseline" | "Regressed" | "Improved" | "Framework";

export interface Experiment {
  id: string;
  name: string;
  variable: string;
  hypothesis: string;
  result: string;
  mean: number | null;
  delta: string | null;
  verdict: Verdict;
  confidence: string;
  details: string;
}

export const EXPERIMENTS: Experiment[] = [
  {
    id: "baseline",
    name: "Baseline PPO",
    variable: "Reference model",
    hypothesis:
      "A PPO agent with a CNN policy can learn competent driving on CarRacing-v3.",
    result: "Mean reward 598.42",
    mean: 598.42,
    delta: null,
    verdict: "Baseline",
    confidence: "95% CI [544.8, 652.1]",
    details:
      "Trained for 1,000,000 steps (CnnPolicy, lr 3e-4, ent_coef 0.005). Evaluated over 30 deterministic episodes on fixed seeds 1000–1029. This is the frozen reference every experiment is measured against; it drives competently but does not solve the track (0% success at reward 900).",
  },
  {
    id: "continue",
    name: "Continue Training",
    variable: "+1M → 3M continued steps",
    hypothesis:
      "Training the converged 1M model further will keep improving it.",
    result: "Mean reward 420.94  (−29.7%)",
    mean: 420.94,
    delta: "−29.7%",
    verdict: "Regressed",
    confidence: "Non-overlapping 95% CIs",
    details:
      "Continuing to train the converged policy at the full constant learning rate drifted it away from its good solution — the classic late-PPO variance. More steps did not mean a better agent; the hypothesis was rejected.",
  },
  {
    id: "lr",
    name: "Learning-Rate Schedule",
    variable: "Constant → linear-decay LR",
    hypothesis:
      "A linear learning-rate decay stabilises late training and avoids the regression.",
    result: "Mean reward 434.10  (−27.5%)",
    mean: 434.1,
    delta: "−27.5%",
    verdict: "Regressed",
    confidence: "Paired t = −5.50 (df 29)",
    details:
      "The regression appeared in the first evaluation after resuming — before decay had any effect — so the cause is continuation from a converged checkpoint, not the schedule. Decaying the LR did not rescue it. Hypothesis rejected.",
  },
  {
    id: "framestack",
    name: "RGB FrameStack",
    variable: "1 frame → 4 stacked frames",
    hypothesis:
      "Four stacked frames improve driving because CarRacing is partially observable.",
    result: "Mean reward 411.07  (−31.3%)",
    mean: 411.07,
    delta: "−31.3%",
    verdict: "Regressed",
    confidence: "Cohen's d = −1.16 (large)",
    details:
      "FrameStack(4) quadrupled the input to 12 channels and trained ~5× slower; evaluation reward was still rising at 1M steps — i.e. undertrained at the fixed budget. The result is confounded by channel count and compute, so the temporal hypothesis was not cleanly isolated. Rejected at this budget.",
  },
  {
    id: "reward",
    name: "Reward Framework",
    variable: "Modular reward composition",
    hypothesis:
      "A modular, composable reward system enables controlled reward research.",
    result: "Framework complete & verified",
    mean: null,
    delta: null,
    verdict: "Framework",
    confidence: "Ablations staged",
    details:
      "Reward is composed from independent, YAML-configurable components (progress, speed, centerline, smooth-steering, off-track, collision, idle). Reward shaping applies to training only, so evaluation stays baseline-comparable. Ablation experiments are implemented and verified; full-budget runs are pending.",
  },
];

export interface Finding {
  finding: string;
  evidence: string;
  interpretation: string;
}

export const FINDINGS: Finding[] = [
  {
    finding: "Continued training regressed.",
    evidence: "598.42 → 420.94 (−29.7%); non-overlapping 95% CIs.",
    interpretation:
      "Training a converged PPO policy further at the full learning rate perturbs it downward. More steps is not a free win.",
  },
  {
    finding: "The learning-rate schedule regressed.",
    evidence: "598.42 → 434.10 (−27.5%); paired t = −5.50.",
    interpretation:
      "The drop preceded any decay, so it stems from continuation, not the schedule. You cannot improve this baseline by resuming it.",
  },
  {
    finding: "RGB frame stacking regressed.",
    evidence: "598.42 → 411.07 (−31.3%); Cohen's d = −1.16 (large).",
    interpretation:
      "Four frames meant 12 input channels and ~5× slower training; the agent was still improving at 1M — undertrained, and confounded by channel count.",
  },
];

export interface Tech {
  name: string;
  note: string;
}

export const TECH: Tech[] = [
  { name: "Python", note: "3.12" },
  { name: "PyTorch", note: "policy network" },
  { name: "Stable-Baselines3", note: "PPO" },
  { name: "Gymnasium", note: "CarRacing-v3" },
  { name: "NumPy", note: "arrays" },
  { name: "OpenCV", note: "frames" },
  { name: "FastAPI", note: "inference API" },
  { name: "Next.js", note: "App Router" },
  { name: "TailwindCSS", note: "UI" },
];

export const ARCHITECTURE = [
  { label: "Simulator", note: "CarRacing-v3 environment" },
  { label: "Environment", note: "factory · single creation point" },
  { label: "Wrappers", note: "observation · reward · action · monitor" },
  { label: "PPO Agent", note: "BaseAgent → PPOAgent (SB3)" },
  { label: "Evaluation", note: "30 episodes · 95% CI · paired t-test" },
  { label: "Research", note: "specs · comparisons · papers" },
];
