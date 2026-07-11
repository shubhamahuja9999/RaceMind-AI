import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "RaceMind AI Research Lab",
  description:
    "A Modular Reinforcement Learning Research Platform for Autonomous Racing. Watch the trained PPO agent, compare it to a random policy, and explore the experiments.",
  keywords: [
    "reinforcement learning",
    "PPO",
    "CarRacing",
    "Stable-Baselines3",
    "research",
    "machine learning",
  ],
  openGraph: {
    title: "RaceMind AI Research Lab",
    description:
      "A Modular Reinforcement Learning Research Platform for Autonomous Racing.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="font-sans">{children}</body>
    </html>
  );
}
