"use client";

import { animate, motion, useMotionValue, useTransform } from "framer-motion";
import { useEffect } from "react";

/** Counts up to `value` when it changes (subtle number-counter animation). */
export function AnimatedNumber({
  value,
  decimals = 0,
}: {
  value: number;
  decimals?: number;
}) {
  const count = useMotionValue(0);
  const text = useTransform(count, (v) =>
    v.toLocaleString("en-US", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    })
  );

  useEffect(() => {
    const controls = animate(count, value, { duration: 0.9, ease: "easeOut" });
    return () => controls.stop();
  }, [count, value]);

  return <motion.span>{text}</motion.span>;
}
