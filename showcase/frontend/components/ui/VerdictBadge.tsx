import { CheckCircle2, XCircle, Layers, TrendingUp } from "lucide-react";
import type { Verdict } from "@/lib/data";

const STYLES: Record<Verdict, { cls: string; icon: typeof CheckCircle2 }> = {
  Baseline: { cls: "bg-brand-50 text-brand-700 border-brand-200", icon: CheckCircle2 },
  Improved: { cls: "bg-emerald-50 text-emerald-700 border-emerald-200", icon: TrendingUp },
  Regressed: { cls: "bg-rose-50 text-rose-700 border-rose-200", icon: XCircle },
  Framework: { cls: "bg-slate-100 text-slate-600 border-slate-200", icon: Layers },
};

export function VerdictBadge({ verdict }: { verdict: Verdict }) {
  const { cls, icon: Icon } = STYLES[verdict];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${cls}`}
    >
      <Icon className="h-3.5 w-3.5" />
      {verdict}
    </span>
  );
}
