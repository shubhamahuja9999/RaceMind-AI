import { Github } from "lucide-react";
import { LINKS } from "@/lib/data";

export function Footer() {
  return (
    <footer className="border-t border-slate-100 py-10">
      <div className="section flex flex-col items-center justify-between gap-4 text-sm text-slate-500 sm:flex-row">
        <p>
          RaceMind AI Research Lab — a reproducible RL research platform for
          <span className="text-slate-700"> CarRacing-v3</span>.
        </p>
        <div className="flex items-center gap-4">
          <span>MIT License</span>
          <a
            href={LINKS.github}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 transition hover:text-slate-900"
          >
            <Github className="h-4 w-4" /> GitHub
          </a>
        </div>
      </div>
    </footer>
  );
}
