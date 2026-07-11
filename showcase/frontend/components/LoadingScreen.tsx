"use client";

import { useEffect, useState } from "react";
import { Cpu } from "lucide-react";

export function LoadingScreen() {
  const [visible, setVisible] = useState(true);
  const [fadeOut, setFadeOut] = useState(false);

  useEffect(() => {
    // Start fade-out after content is likely painted
    const timer = setTimeout(() => {
      setFadeOut(true);
      // Remove from DOM after animation completes
      setTimeout(() => setVisible(false), 600);
    }, 1200);

    return () => clearTimeout(timer);
  }, []);

  if (!visible) return null;

  return (
    <div
      className={`fixed inset-0 z-[9999] flex flex-col items-center justify-center bg-slate-950 transition-opacity duration-500 ${
        fadeOut ? "opacity-0" : "opacity-100"
      }`}
    >
      {/* Animated logo */}
      <div className="relative mb-8 flex h-16 w-16 items-center justify-center">
        {/* Outer ring pulse */}
        <div className="absolute inset-0 animate-ping rounded-2xl bg-brand-500/20" />
        {/* Inner glow */}
        <div className="absolute inset-0 animate-pulse rounded-2xl bg-brand-500/10" />
        {/* Logo */}
        <div className="relative flex h-16 w-16 items-center justify-center rounded-2xl bg-brand-600 shadow-lg shadow-brand-500/30">
          <Cpu className="h-8 w-8 text-white" />
        </div>
      </div>

      {/* Title */}
      <h1 className="mb-2 text-2xl font-bold tracking-tight text-white">
        RaceMind AI
      </h1>

      {/* Subtitle */}
      <p className="mb-8 text-sm text-slate-400">
        Interactive Reinforcement Learning Research Lab
      </p>

      {/* Loading bar */}
      <div className="h-1 w-48 overflow-hidden rounded-full bg-slate-800">
        <div className="loading-bar h-full rounded-full bg-gradient-to-r from-brand-500 via-brand-400 to-brand-500" />
      </div>

      {/* Status text */}
      <p className="mt-4 text-xs text-slate-500">Loading model...</p>
    </div>
  );
}
