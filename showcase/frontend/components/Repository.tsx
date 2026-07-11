"use client";

import { ArrowUpRight, BookOpen, FileText, Github, Tag } from "lucide-react";
import { LINKS } from "@/lib/data";
import { Section } from "./ui/Section";
import { Reveal } from "./ui/Reveal";

const CARDS = [
  { icon: Github, title: "GitHub", desc: "Source code, packages and experiments.", href: LINKS.github },
  { icon: Tag, title: "Release v1.0", desc: "Tagged release with changelog.", href: LINKS.release },
  { icon: BookOpen, title: "Documentation", desc: "Architecture, training, evaluation guides.", href: LINKS.docs },
  { icon: FileText, title: "Research Report", desc: "Methodology, results and discussion.", href: LINKS.report },
];

export function Repository() {
  return (
    <Section id="repo" eyebrow="08 · Repository" title="Explore the project">
      <div className="mx-auto grid max-w-4xl gap-4 sm:grid-cols-2">
        {CARDS.map((c, i) => (
          <Reveal key={c.title} delay={i * 0.06}>
            <a
              href={c.href}
              target="_blank"
              rel="noreferrer"
              className="card group flex items-center gap-4 p-5"
            >
              <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-brand-50 text-brand-600">
                <c.icon className="h-5 w-5" />
              </span>
              <div className="flex-1">
                <div className="flex items-center gap-1 font-semibold text-slate-900">
                  {c.title}
                  <ArrowUpRight className="h-4 w-4 text-slate-300 transition group-hover:text-brand-500" />
                </div>
                <p className="mt-0.5 text-sm text-slate-500">{c.desc}</p>
              </div>
            </a>
          </Reveal>
        ))}
      </div>
    </Section>
  );
}
