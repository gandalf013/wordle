import type { GuessAnalysis } from './analysis';

export function skillScore(guessEntropy: number, allEntropies: number[]): number {
  const n = allEntropies.length;
  if (n === 0) return 0;
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (allEntropies[i] <= guessEntropy + 1e-9) {
      count++;
    }
  }
  return Math.round((100.0 * count) / n);
}

export function luckScore(bucketSizes: number[], actualBucketSize: number): number {
  let n = 0;
  for (const s of bucketSizes) n += s;
  if (n <= 0 || actualBucketSize <= 0) return 0;

  let kMax = 0;
  for (const s of bucketSizes) {
    if (s > kMax) kMax = s;
  }

  const k = actualBucketSize;
  const infoActual = Math.log2(n) - Math.log2(k);
  const infoMax = Math.log2(n); // k == 1
  const infoMin = Math.log2(n) - Math.log2(kMax);

  if (Math.abs(infoMax - infoMin) < 1e-9) {
    return k === 1 ? 100 : 0;
  }

  const luck = (100.0 * (infoActual - infoMin)) / (infoMax - infoMin);
  return Math.round(Math.min(100.0, Math.max(0.0, luck)));
}

export function calculateSkill(analysis: GuessAnalysis, allAnalyses: GuessAnalysis[]): number {
  return skillScore(analysis.entropy, allAnalyses.map(a => a.entropy));
}

export function calculateLuck(analysis: GuessAnalysis, actualScore: number): number {
  if (analysis.buckets) {
    const bucket = analysis.buckets[actualScore];
    if (!bucket) return 0.0;
    const sizes = Object.values(analysis.buckets).map(words => words.length);
    return luckScore(sizes, bucket.length);
  }
  if (analysis.bucketCounts) {
    const found = analysis.bucketCounts.find(bc => bc.score === actualScore);
    if (!found) return 0.0;
    const sizes = analysis.bucketCounts.map(bc => bc.count);
    return luckScore(sizes, found.count);
  }
  return 0.0;
}
