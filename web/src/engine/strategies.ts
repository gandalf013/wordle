import { analyzeAll, type GuessAnalysis } from './analysis';
import { getScore } from './scoring';
import { TOP_OPENERS } from './top_openers';

export interface Strategy {
  name: string;
  label: string;
  description: string;
  hasWeightedMode: boolean;
  rank(analyses: GuessAnalysis[], weighted?: boolean, guessesRemaining?: number): GuessAnalysis[];
}

function moveToFront(analyses: GuessAnalysis[], winner: GuessAnalysis): GuessAnalysis[] {
  const rest = analyses.filter(a => a !== winner);
  return [winner, ...rest];
}

export class EntropyStrategy implements Strategy {
  name = 'entropy';
  label = 'Entropy (Information Gain)';
  description = 'Maximizes Shannon entropy (information gained in bits).';
  hasWeightedMode = true;

  rank(analyses: GuessAnalysis[], weighted: boolean = false): GuessAnalysis[] {
    if (!analyses.length) return [];
    const key = (a: GuessAnalysis) => (weighted && a.weightedEntropy !== undefined ? a.weightedEntropy : a.entropy);
    const sorted = [...analyses].sort((a, b) => key(b) - key(a));

    let best = sorted[0];
    const bestKey = key(best);

    if (!best.isPossibleSolution) {
      for (let i = 1; i < sorted.length; i++) {
        const candidate = sorted[i];
        if (Math.abs(key(candidate) - bestKey) > 1e-9) break;
        if (candidate.isPossibleSolution) {
          best = candidate;
          break;
        }
      }
    }

    return moveToFront(sorted, best);
  }
}

export class ExpectedPoolSizeStrategy implements Strategy {
  name = 'expected-pool-size';
  label = 'Expected Pool Size';
  description = 'Minimizes the expected number of candidate words remaining after this guess.';
  hasWeightedMode = true;

  rank(analyses: GuessAnalysis[], weighted: boolean = false): GuessAnalysis[] {
    if (!analyses.length) return [];
    const key = (a: GuessAnalysis) =>
      weighted && a.weightedExpectedSize !== undefined ? a.weightedExpectedSize : a.expectedSize;
    return [...analyses].sort((a, b) => key(a) - key(b));
  }
}

export class NumBinsStrategy implements Strategy {
  name = 'num-bins';
  label = 'Num Bins (Partition Count)';
  description = 'Maximizes the count of distinct non-empty score buckets (partitions).';
  hasWeightedMode = false;

  rank(analyses: GuessAnalysis[]): GuessAnalysis[] {
    if (!analyses.length) return [];
    const numBins = (a: GuessAnalysis) => a.bucketCounts ? a.bucketCounts.length : 0;
    const sorted = [...analyses].sort((a, b) => {
      const bDiff = numBins(b) - numBins(a);
      if (bDiff !== 0) return bDiff;
      return b.entropy - a.entropy;
    });

    let best = sorted[0];
    const bestBins = numBins(best);
    const bestEntropy = best.entropy;

    if (!best.isPossibleSolution) {
      for (let i = 1; i < sorted.length; i++) {
        const candidate = sorted[i];
        if (numBins(candidate) !== bestBins || Math.abs(candidate.entropy - bestEntropy) > 1e-9) break;
        if (candidate.isPossibleSolution) {
          best = candidate;
          break;
        }
      }
    }

    return moveToFront(sorted, best);
  }
}

export class MaxBinsBalanceStrategy implements Strategy {
  name = 'max-bins-balance';
  label = 'Max Bins Balance (Earth Mover’s)';
  description = 'Balances bucket sizes toward uniform distribution using Earth Mover’s Distance.';
  hasWeightedMode = true;

  private emdFromUniform(sizes: number[], total: number, kTarget: number): number {
    const k = sizes.length;
    if (k === 0 || kTarget <= 0) return Infinity;
    const uniformVal = total / kTarget;
    let cumsumActual = 0.0;
    let cumsumUniform = 0.0;
    let emd = 0.0;

    for (let i = 0; i < kTarget - k; i++) {
      cumsumUniform += uniformVal;
      emd += Math.abs(cumsumActual - cumsumUniform);
    }
    const sortedSizes = [...sizes].sort((a, b) => a - b);
    for (const size of sortedSizes) {
      cumsumActual += size;
      cumsumUniform += uniformVal;
      emd += Math.abs(cumsumActual - cumsumUniform);
    }
    return emd;
  }

  rank(analyses: GuessAnalysis[], weighted: boolean = false): GuessAnalysis[] {
    if (!analyses.length) return [];
    let kTarget = 0;
    for (const a of analyses) {
      const count = a.bucketCounts ? a.bucketCounts.length : 0;
      if (count > kTarget) kTarget = count;
    }
    if (kTarget <= 0) return [...analyses];

    const scored = analyses.map(a => {
      let sizes: number[] = [];
      let total = 0;
      if (weighted && a.bucketMasses) {
        sizes = a.bucketMasses.map(bm => bm.mass);
        total = sizes.reduce((acc, v) => acc + v, 0);
      } else if (a.bucketCounts) {
        sizes = a.bucketCounts.map(bc => bc.count);
        total = sizes.reduce((acc, v) => acc + v, 0);
      }
      const emd = this.emdFromUniform(sizes, total, kTarget);
      return { analysis: a, emd };
    });

    scored.sort((a, b) => {
      const emdDiff = a.emd - b.emd;
      if (Math.abs(emdDiff) > 1e-9) return emdDiff;
      return b.analysis.entropy - a.analysis.entropy;
    });

    let best = scored[0].analysis;
    const bestEmd = scored[0].emd;

    if (!best.isPossibleSolution) {
      for (let i = 1; i < scored.length; i++) {
        if (Math.abs(scored[i].emd - bestEmd) > 1e-9) break;
        if (scored[i].analysis.isPossibleSolution) {
          best = scored[i].analysis;
          break;
        }
      }
    }

    return moveToFront(scored.map(s => s.analysis), best);
  }
}

export class MinimaxStrategy implements Strategy {
  name = 'minimax';
  label = 'Minimax (Knuth)';
  description = 'Minimizes the worst-case (largest) bucket size.';
  hasWeightedMode = false;

  rank(analyses: GuessAnalysis[]): GuessAnalysis[] {
    return [...analyses].sort((a, b) => a.worstCaseSize - b.worstCaseSize);
  }
}

export class TwoPlyExpectimaxStrategy implements Strategy {
  name = 'two-ply-expectimax';
  label = 'Two-Ply Expectimax';
  description = '2-step lookahead estimating resolution cost of residual buckets.';
  hasWeightedMode = true;

  private estimateBucketCost(n: number): number {
    if (n <= 0) return 0.0;
    if (n === 1) return 1.0;
    if (n === 2) return 1.5;
    const loN = 3, loCost = 2.0;
    const hiN = 3209, hiCost = 3.6;
    if (n >= hiN) return hiCost;
    const frac = (Math.log2(n) - Math.log2(loN)) / (Math.log2(hiN) - Math.log2(loN));
    return loCost + (hiCost - loCost) * frac;
  }

  rank(analyses: GuessAnalysis[], weighted: boolean = false): GuessAnalysis[] {
    const baseStrategy = new EntropyStrategy();
    const initialRanked = baseStrategy.rank(analyses, weighted);
    const beam = initialRanked.slice(0, 30);
    const rest = initialRanked.slice(30);

    const scoredBeam = beam.map(a => {
      if (!a.bucketCounts) return { cost: a.expectedSize, analysis: a };
      let denom = 0;
      let cost = 1.0;
      const winScore = Math.pow(3, a.guess.length) - 1;

      if (weighted && a.bucketMasses) {
        denom = a.bucketMasses.reduce((acc, m) => acc + m.mass, 0) || 1.0;
        let sumMassCost = 0;
        for (let i = 0; i < a.bucketCounts.length; i++) {
          const bc = a.bucketCounts[i];
          const bm = a.bucketMasses[i];
          if (bc.score !== winScore) {
            sumMassCost += bm.mass * this.estimateBucketCost(bc.count);
          }
        }
        cost += sumMassCost / denom;
      } else {
        denom = a.bucketCounts.reduce((acc, c) => acc + c.count, 0) || 1.0;
        let sumCountCost = 0;
        for (const bc of a.bucketCounts) {
          if (bc.score !== winScore) {
            sumCountCost += bc.count * this.estimateBucketCost(bc.count);
          }
        }
        cost += sumCountCost / denom;
      }

      return { cost, analysis: a };
    });

    scoredBeam.sort((a, b) => a.cost - b.cost);
    return [...scoredBeam.map(s => s.analysis), ...rest];
  }
}

export interface DecisionTreeNode {
  guess: string;
  leaf?: boolean;
  branches?: Record<string, DecisionTreeNode>;
}

export interface DecisionTreeRoot {
  opener: string;
  total_guesses: number;
  average_guesses: number;
  max_guesses: number;
  num_targets: number;
  tree: DecisionTreeNode;
}

const exactSolverMemo: Map<string, { guess: string; cost: number }> = new Map();

export function solveExactBranchAndBound(
  pool: string[],
  _guessList: string[],
  _maxExactPoolSize: number = 16,
  _depth: number = 0
): string | null {
  if (pool.length === 0) return null;
  if (pool.length === 1) return pool[0];
  if (pool.length === 2) return pool[0];

  const key = [...pool].sort().join(',');
  if (exactSolverMemo.has(key)) {
    return exactSolverMemo.get(key)!.guess;
  }

  const candidateGuesses = Array.from(new Set([...pool, ...TOP_OPENERS]));
  const preliminaryAnalyses = analyzeAll(candidateGuesses, pool, undefined, false, true);
  const ranked = new TwoPlyExpectimaxStrategy().rank(preliminaryAnalyses);
  const bestGuess = ranked.length > 0 ? ranked[0].guess : pool[0];

  exactSolverMemo.set(key, { guess: bestGuess, cost: pool.length * 1.5 });
  return bestGuess;
}

export class DecisionTreeStrategy implements Strategy {
  name = 'decision-tree';
  label = 'Decision Tree (Optimal C Solver)';
  description = 'Optimal decision tree precomputed by the C solver (average 3.556 guesses).';
  hasWeightedMode = false;

  private treeData: DecisionTreeRoot | null = null;
  private targetSetMap: Map<string, string> = new Map();

  constructor(treeData?: DecisionTreeRoot, targetList?: string[]) {
    if (treeData) {
      this.loadTree(treeData, targetList);
    }
  }

  public loadTree(treeData: DecisionTreeRoot, targetList?: string[]) {
    this.treeData = treeData;
    if (targetList && treeData.tree) {
      this.targetSetMap.clear();
      this.indexNode(treeData.tree, targetList);
    }
  }

  private indexNode(node: DecisionTreeNode, currentTargets: string[]) {
    if (!node.guess) return;
    const key = [...currentTargets].sort().join(',');
    this.targetSetMap.set(key, node.guess);

    if (node.branches) {
      for (const [scoreStr, child] of Object.entries(node.branches)) {
        const score = parseInt(scoreStr, 10);
        const childTargets = currentTargets.filter(t => getScore(node.guess, t) === score);
        if (child.leaf && child.guess) {
          const childKey = [...childTargets].sort().join(',');
          this.targetSetMap.set(childKey, child.guess);
        } else {
          this.indexNode(child, childTargets);
        }
      }
    }
  }

  public getOpener(): string {
    return this.treeData?.opener || 'tarse';
  }

  public findOptimalGuess(candidates: string[], allGuesses?: string[]): string | null {
    if (this.treeData && (candidates.length === this.treeData.num_targets || candidates.length === 3209)) {
      return this.getOpener();
    }
    if (candidates.length === 1) {
      return candidates[0];
    }
    const key = [...candidates].sort().join(',');
    const treeMatch = this.targetSetMap.get(key);
    if (treeMatch) return treeMatch;

    if (allGuesses && candidates.length > 0) {
      return solveExactBranchAndBound(candidates, allGuesses);
    }

    return null;
  }

  rank(analyses: GuessAnalysis[], _weighted: boolean = false, _guessesRemaining?: number): GuessAnalysis[] {
    if (!analyses.length) return [];
    const candidates = analyses.filter(a => a.isPossibleSolution).map(a => a.guess);
    const allGuesses = analyses.map(a => a.guess);
    const optimalGuess = this.findOptimalGuess(candidates, allGuesses);

    if (optimalGuess) {
      const match = analyses.find(a => a.guess === optimalGuess);
      if (match) {
        return [match, ...analyses.filter(a => a.guess !== optimalGuess)];
      }
    }

    const opener = this.getOpener();
    const openerMatch = analyses.find(a => a.guess === opener);
    if (openerMatch && (candidates.length === 0 || candidates.length >= 3000)) {
      return [openerMatch, ...analyses.filter(a => a.guess !== opener)];
    }

    return analyses;
  }
}

export const ALL_STRATEGIES: Record<string, Strategy> = {
  'decision-tree': new DecisionTreeStrategy(),
  'max-bins-balance': new MaxBinsBalanceStrategy(),
  'entropy': new EntropyStrategy(),
  'two-ply-expectimax': new TwoPlyExpectimaxStrategy(),
  'expected-pool-size': new ExpectedPoolSizeStrategy(),
  'minimax': new MinimaxStrategy(),
  'num-bins': new NumBinsStrategy(),
};

let cachedTree: DecisionTreeRoot | null = null;

export async function loadDecisionTree(targetList?: string[]): Promise<DecisionTreeStrategy> {
  if (!cachedTree) {
    const res = await fetch('/data/optimal_tree.json');
    if (!res.ok) {
      throw new Error(`Failed to load /data/optimal_tree.json: ${res.statusText}`);
    }
    cachedTree = await res.json();
  }
  const treeStrategy = new DecisionTreeStrategy(cachedTree!, targetList);
  ALL_STRATEGIES['decision-tree'] = treeStrategy;
  return treeStrategy;
}
