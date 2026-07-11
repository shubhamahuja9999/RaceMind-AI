import type { ReactNode } from "react";
import { Reveal } from "./Reveal";

/** A page section with a consistent eyebrow, title, and lead paragraph. */
export function Section({
  id,
  eyebrow,
  title,
  lead,
  children,
  className = "",
}: {
  id?: string;
  eyebrow?: string;
  title: string;
  lead?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section id={id} className={`scroll-mt-20 py-16 sm:py-24 ${className}`}>
      <div className="section">
        <Reveal>
          <div className="mx-auto max-w-2xl text-center">
            {eyebrow && <span className="eyebrow">{eyebrow}</span>}
            <h2 className="mt-4 text-3xl font-semibold tracking-tight text-slate-900 sm:text-4xl">
              {title}
            </h2>
            {lead && <p className="mt-4 text-lg leading-relaxed text-slate-600">{lead}</p>}
          </div>
        </Reveal>
        <div className="mt-12">{children}</div>
      </div>
    </section>
  );
}
