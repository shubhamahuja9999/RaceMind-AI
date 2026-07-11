"use client";

import { Github, Cpu } from "lucide-react";
import { LINKS } from "@/lib/data";

const NAV = [
  { href: "#overview", label: "Overview" },
  { href: "#timeline", label: "Experiments" },
  { href: "#run", label: "Run AI" },
  { href: "#dashboard", label: "Dashboard" },
  { href: "#findings", label: "Findings" },
];

export function Navbar() {
  return (
    <nav className="sticky top-0 z-50 border-b border-slate-100 bg-white/80 backdrop-blur">
      <div className="section flex h-14 items-center justify-between">
        <a href="#top" className="flex items-center gap-2 font-semibold text-slate-900">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-brand-600 text-white">
            <Cpu className="h-4 w-4" />
          </span>
          RaceMind AI
        </a>
        <div className="hidden items-center gap-6 md:flex">
          {NAV.map((item) => (
            <a
              key={item.href}
              href={item.href}
              className="text-sm text-slate-500 transition hover:text-slate-900"
            >
              {item.label}
            </a>
          ))}
        </div>
        <a
          href={LINKS.github}
          target="_blank"
          rel="noreferrer"
          aria-label="GitHub repository"
          className="inline-flex items-center gap-2 rounded-lg border border-slate-200 px-3 py-1.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
        >
          <Github className="h-4 w-4" /> <span className="hidden sm:inline">GitHub</span>
        </a>
      </div>
    </nav>
  );
}
