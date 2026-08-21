import { getScore } from './scoring';

export interface BucketCounts {
  score: number;
  count: number;
}

export interface BucketMasses {
  score: number;
  mass: number;
}

export interface GuessAnalysis {
  guess: string;
  entropy: number;
  worstCaseSize: number;
  expectedSize: number;
  isPossibleSolution: boolean;
  buckets?: Record<number, string[]>;
  weightedEntropy?: number;
  weightedExpectedSize?: number;
  solutionProbability?: number;
  bucketCounts?: BucketCounts[];
  bucketMasses?: BucketMasses[];
}

export function computeEntropy(sizes: number[], total: number): number {
  if (total <= 0) return 0.0;
  let ent = 0.0;
  for (const s of sizes) {
    if (s > 0) {
      const p = s / total;
      ent -= p * Math.log2(p);
    }
  }
  return ent;
}

export function analyze(
  guess: string,
  targetPool: string[],
  weights?: Record<string, number>,
  includeBuckets: boolean = true
): GuessAnalysis {
  const buckets: Record<number, string[]> = {};
  const targetSet = new Set(targetPool);
  const isPossibleSolution = targetSet.has(guess);

  for (let i = 0; i < targetPool.length; i++) {
    const target = targetPool[i];
    const score = getScore(guess, target);
    if (!buckets[score]) {
      buckets[score] = [];
    }
    buckets[score].push(target);
  }

  const scores = Object.keys(buckets).map(Number).sort((a, b) => a - b);
  const counts = scores.map(s => buckets[s].length);
  const total = targetPool.length;

  const entropy = computeEntropy(counts, total);
  const worstCaseSize = counts.length > 0 ? Math.max(...counts) : 0;
  
  let sumSq = 0;
  for (const c of counts) {
    sumSq += c * c;
  }
  const expectedSize = total > 0 ? sumSq / total : 0;

  const bucketCounts: BucketCounts[] = scores.map(s => ({
    score: s,
    count: buckets[s].length,
  }));

  let weightedEntropy: number | undefined;
  let weightedExpectedSize: number | undefined;
  let solutionProbability: number | undefined;
  let bucketMasses: BucketMasses[] | undefined;

  if (weights) {
    let totalMass = 0;
    const masses: number[] = [];
    const bMasses: BucketMasses[] = [];

    for (const s of scores) {
      const words = buckets[s];
      let m = 0;
      for (const w of words) {
        m += weights[w] !== undefined ? weights[w] : 1.0;
      }
      masses.push(m);
      totalMass += m;
      bMasses.push({ score: s, mass: m });
    }

    bucketMasses = bMasses;

    if (totalMass > 0) {
      weightedEntropy = computeEntropy(masses, totalMass);
      let wExp = 0;
      for (let i = 0; i < masses.length; i++) {
        wExp += (masses[i] / totalMass) * counts[i];
      }
      weightedExpectedSize = wExp;
      const guessWeight = isPossibleSolution && weights[guess] !== undefined ? weights[guess] : 1.0;
      solutionProbability = isPossibleSolution ? guessWeight / totalMass : 0.0;
    } else {
      weightedEntropy = 0.0;
      weightedExpectedSize = 0.0;
      solutionProbability = 0.0;
    }
  }

  return {
    guess,
    entropy,
    worstCaseSize,
    expectedSize,
    isPossibleSolution,
    buckets: includeBuckets ? buckets : undefined,
    weightedEntropy,
    weightedExpectedSize,
    solutionProbability,
    bucketCounts,
    bucketMasses,
  };
}

const MAX_SCORES = 243;
const countBuf = new Int32Array(MAX_SCORES);
const massBuf = new Float64Array(MAX_SCORES);
const activeScoresBuf = new Int32Array(MAX_SCORES);

export function analyzeAll(
  guessList: string[],
  targetPool: string[],
  weights?: Record<string, number>,
  includeBuckets: boolean = false,
  includeBucketStats: boolean = true
): GuessAnalysis[] {
  if (!guessList.length || !targetPool.length) return [];
  const targetSet = new Set(targetPool);
  const G = guessList.length;
  const T = targetPool.length;
  const invT = 1.0 / T;

  const results: GuessAnalysis[] = new Array(G);

  for (let g = 0; g < G; g++) {
    const guess = guessList[g];
    const isPossibleSolution = targetSet.has(guess);

    countBuf.fill(0);
    if (weights) massBuf.fill(0);
    let numActive = 0;

    let bucketWordMap: Record<number, string[]> | undefined;
    if (includeBuckets) {
      bucketWordMap = {};
    }

    for (let t = 0; t < T; t++) {
      const target = targetPool[t];
      const s = getScore(guess, target);
      if (countBuf[s] === 0) {
        activeScoresBuf[numActive++] = s;
      }
      countBuf[s]++;

      if (weights) {
        const w = weights[target] !== undefined ? weights[target] : 1.0;
        massBuf[s] += w;
      }
      if (includeBuckets) {
        if (!bucketWordMap![s]) bucketWordMap![s] = [];
        bucketWordMap![s].push(target);
      }
    }

    let entropy = 0.0;
    let worstCase = 0;
    let sumSq = 0;

    for (let i = 0; i < numActive; i++) {
      const s = activeScoresBuf[i];
      const c = countBuf[s];
      if (c > worstCase) worstCase = c;
      sumSq += c * c;
      const p = c * invT;
      entropy -= p * Math.log2(p);
    }
    const expectedSize = sumSq * invT;

    let bucketCounts: BucketCounts[] | undefined;
    let bucketMasses: BucketMasses[] | undefined;
    let weightedEntropy: number | undefined;
    let weightedExpectedSize: number | undefined;
    let solutionProbability: number | undefined;

    if (includeBucketStats) {
      bucketCounts = new Array(numActive);
      for (let i = 0; i < numActive; i++) {
        const s = activeScoresBuf[i];
        bucketCounts[i] = { score: s, count: countBuf[s] };
      }
    }

    if (weights) {
      let totalMass = 0;
      for (let i = 0; i < numActive; i++) {
        totalMass += massBuf[activeScoresBuf[i]];
      }

      if (includeBucketStats) {
        bucketMasses = new Array(numActive);
        for (let i = 0; i < numActive; i++) {
          const s = activeScoresBuf[i];
          bucketMasses[i] = { score: s, mass: massBuf[s] };
        }
      }

      if (totalMass > 0) {
        const invMass = 1.0 / totalMass;
        let wEnt = 0.0;
        let wExp = 0.0;
        for (let i = 0; i < numActive; i++) {
          const s = activeScoresBuf[i];
          const m = massBuf[s];
          const c = countBuf[s];
          const p = m * invMass;
          if (p > 0) wEnt -= p * Math.log2(p);
          wExp += p * c;
        }
        weightedEntropy = wEnt;
        weightedExpectedSize = wExp;
        const guessW = isPossibleSolution && weights[guess] !== undefined ? weights[guess] : 1.0;
        solutionProbability = isPossibleSolution ? guessW * invMass : 0.0;
      }
    }

    results[g] = {
      guess,
      entropy,
      worstCaseSize: worstCase,
      expectedSize,
      isPossibleSolution,
      buckets: bucketWordMap,
      weightedEntropy,
      weightedExpectedSize,
      solutionProbability,
      bucketCounts,
      bucketMasses,
    };
  }

  return results;
}
