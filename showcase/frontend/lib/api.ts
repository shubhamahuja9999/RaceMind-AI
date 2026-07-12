// API client for the RaceMind AI Research Lab backend.

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

export interface EpisodeResult {
  policy: string;
  seed: number;
  reward: number;
  length: number;
  inference_seconds: number;
  steps_per_second: number;
  video_url: string; // relative to API_URL, e.g. /videos/best_1028.mp4
}

/** Absolute URL for a video returned by the backend. */
export function videoSrc(result: EpisodeResult): string {
  return `${API_URL}${result.video_url}`;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Fetch that transparently waits out transient backend states:
 *  - 429: another inference is running (backend allows one at a time)
 *  - 503: model still loading (cold start)
 * Both resolve on their own within seconds, so we retry rather than surfacing an
 * error the user can't act on.
 */
async function fetchWithRetry(
  url: string,
  init?: RequestInit,
  { retries = 10, delayMs = 4000 } = {}
): Promise<Response> {
  for (let attempt = 0; ; attempt++) {
    const res = await fetch(url, init);
    if ((res.status === 429 || res.status === 503) && attempt < retries) {
      await sleep(delayMs);
      continue;
    }
    return res;
  }
}

export async function checkHealth(): Promise<{ status: string; model_loaded: boolean }> {
  return json(await fetch(`${API_URL}/api/health`, { cache: "no-store" }));
}

export async function runPolicy(
  policy: "best" | "random",
  seed = 1028
): Promise<EpisodeResult> {
  const res = await fetchWithRetry(`${API_URL}/api/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ policy, seed }),
  });
  return json<EpisodeResult>(res);
}
