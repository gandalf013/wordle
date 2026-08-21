import { getScore, getScoreList, Score, parseScoreString } from './scoring';
import type { WordList } from './wordlists';
import { analyze, analyzeAll } from './analysis';
import type { GuessAnalysis } from './analysis';
import { ALL_STRATEGIES, DecisionTreeStrategy } from './strategies';
import type { Strategy } from './strategies';
import { TOP_OPENERS } from './top_openers';
import { MoveTree, type MoveNode } from './tree';

export type { MoveNode };

export class GameEngine {
  public wordList: WordList;
  public activeStrategy: Strategy;
  public mode: 'known' | 'unknown' = 'known';
  public isWeighted: boolean = false;
  public secretSolution: string | null = null;
  public hideSecret: boolean = true;
  public showOptimalCard: boolean = false;
  public tree: MoveTree;

  // Listeners for UI reactive re-renders
  private listeners: (() => void)[] = [];

  constructor(wordList: WordList, strategy: Strategy = ALL_STRATEGIES['decision-tree']) {
    this.wordList = wordList;
    this.activeStrategy = strategy;
    this.tree = new MoveTree(wordList.target);
    this.setRandomSecret();
  }

  public subscribe(fn: () => void) {
    this.listeners.push(fn);
    return () => {
      this.listeners = this.listeners.filter(l => l !== fn);
    };
  }

  public notify() {
    for (const fn of this.listeners) fn();
  }

  public get activePath(): MoveNode[] {
    return this.tree.getActivePath();
  }

  public get activeNode(): MoveNode | null {
    return this.tree.activeNode;
  }

  public get currentCandidates(): string[] {
    return this.tree.getActiveCandidates();
  }

  public get isInconsistent(): boolean {
    return this.tree.activeNode !== null && this.currentCandidates.length === 0;
  }

  public setMode(mode: 'known' | 'unknown') {
    this.mode = mode;
    if (mode === 'known') {
      if (!this.secretSolution) {
        this.setRandomSecret();
      }
    }
    this.notify();
  }

  public setSecretSolution(solution: string | null, resetGame: boolean = true) {
    this.secretSolution = solution ? solution.toLowerCase().trim() : null;
    if (this.secretSolution) {
      this.mode = 'known';
    }
    if (resetGame) {
      this.reset(false);
    } else {
      this.notify();
    }
  }

  public setRandomSecret(resetGame: boolean = true) {
    const randomIndex = Math.floor(Math.random() * this.wordList.target.length);
    const word = this.wordList.target[randomIndex];
    this.setSecretSolution(word, resetGame);
  }

  public toggleHideSecret() {
    this.hideSecret = !this.hideSecret;
    this.notify();
  }

  public toggleShowOptimalCard() {
    this.showOptimalCard = !this.showOptimalCard;
    this.notify();
  }

  public setStrategy(strategyName: string) {
    if (ALL_STRATEGIES[strategyName]) {
      this.activeStrategy = ALL_STRATEGIES[strategyName];
      this.tree.recomputeAll(this.wordList, this.activeStrategy, this.isWeighted);
      this.notify();
    }
  }

  public setWeighted(weighted: boolean) {
    this.isWeighted = weighted;
    this.tree.recomputeAll(this.wordList, this.activeStrategy, this.isWeighted);
    this.notify();
  }

  public addGuess(guess: string, scoreOverride?: number): MoveNode | null {
    if (this.activePath.length >= 6) {
      return null;
    }
    if (this.activeNode && this.activeNode.score === 242) {
      return null;
    }

    const cleanGuess = guess.trim().toLowerCase();
    if (cleanGuess.length !== this.wordList.wordLength) {
      throw new Error(`Word length must be ${this.wordList.wordLength}`);
    }

    // Prevent duplicate guess along the current active branch
    if (this.activePath.some(n => n.guess === cleanGuess)) {
      return null;
    }

    let score = scoreOverride;
    if (score === undefined) {
      if (this.mode === 'known' && this.secretSolution) {
        score = getScore(cleanGuess, this.secretSolution);
      } else {
        score = 0; // Default all gray in unknown mode
      }
    }

    const node = this.tree.addMove(
      cleanGuess,
      score,
      this.wordList,
      this.activeStrategy,
      this.isWeighted
    );
    this.notify();
    return node;
  }

  public playOptimalMove(): MoveNode | null {
    if (this.activePath.length >= 6) return null;
    if (this.activeNode && this.activeNode.score === 242) return null;
    const opt = this.getSuggestedNextGuess();
    if (!opt) return null;
    return this.addGuess(opt.guess);
  }

  public selectNode(nodeId: string | null) {
    this.tree.selectNode(nodeId);
    this.notify();
  }

  public stepBack(): boolean {
    const ok = this.tree.stepBack();
    if (ok) this.notify();
    return ok;
  }

  public stepForward(): boolean {
    const ok = this.tree.stepForward();
    if (ok) this.notify();
    return ok;
  }

  public promoteToMainLine(nodeId: string) {
    this.tree.promoteToMainLine(nodeId);
    this.notify();
  }

  public deleteNode(nodeId: string) {
    this.tree.deleteNode(nodeId);
    this.notify();
  }

  public updateTileColor(nodeId: string, letterIndex: number) {
    const node = this.tree.nodes.get(nodeId);
    if (!node || letterIndex < 0 || letterIndex >= this.wordList.wordLength) return;

    const currentScores = getScoreList(node.score, this.wordList.wordLength);
    const curr = currentScores[letterIndex];
    // Cycle: GRAY (0) -> YELLOW (1) -> GREEN (2) -> GRAY (0)
    currentScores[letterIndex] = ((curr + 1) % 3) as Score;

    let newScore = 0;
    for (let i = 0; i < this.wordList.wordLength; i++) {
      newScore += Math.pow(3, this.wordList.wordLength - i - 1) * currentScores[i];
    }

    this.tree.updateNodeScore(nodeId, newScore, this.wordList, this.activeStrategy, this.isWeighted);
    this.notify();
  }

  public reset(resetSecret: boolean = false) {
    this.tree.reset();
    this.showOptimalCard = false;
    if (resetSecret && this.mode === 'known') {
      this.setRandomSecret(false);
    }
    this.notify();
  }

  public getSuggestedNextGuess(): GuessAnalysis | null {
    const pool = this.currentCandidates;
    if (pool.length === 0) return null;
    const playedGuesses = new Set(this.activePath.map(n => n.guess));

    if (pool.length === 1) {
      if (playedGuesses.has(pool[0])) return null;
      return analyze(pool[0], pool, this.wordList.weights, true);
    }
    const weights = this.isWeighted && this.activeStrategy.hasWeightedMode ? this.wordList.weights : undefined;

    if (this.activeStrategy instanceof DecisionTreeStrategy) {
      if (pool.length === this.wordList.target.length) {
        const opener = this.activeStrategy.getOpener();
        if (!playedGuesses.has(opener)) {
          return analyze(opener, pool, weights, true);
        }
      }
      const opt = (this.activeStrategy as DecisionTreeStrategy).findOptimalGuess(pool, this.wordList.allGuesses);
      if (opt && !playedGuesses.has(opt)) {
        return analyze(opt, pool, weights, true);
      }
    }

    const evalStrategy = (this.activeStrategy instanceof DecisionTreeStrategy)
      ? ALL_STRATEGIES['entropy']
      : this.activeStrategy;

    const guessesToEvaluate = pool.length <= 500 ? this.wordList.allGuesses : TOP_OPENERS;
    const unplayedGuesses = guessesToEvaluate.filter(g => !playedGuesses.has(g));
    if (unplayedGuesses.length === 0) return null;

    const analyses = analyzeAll(unplayedGuesses, pool, weights, true, true);
    const ranked = evalStrategy.rank(analyses, this.isWeighted);
    return ranked.find(a => !playedGuesses.has(a.guess)) || null;
  }

  public getKeyboardKeyStates(): Record<string, Score> {
    const states: Record<string, Score> = {};
    const path = this.activePath;
    for (const node of path) {
      for (let i = 0; i < node.guess.length; i++) {
        const ch = node.guess[i].toUpperCase();
        const sc = node.scoreList[i];
        if (states[ch] === undefined || sc > states[ch]) {
          states[ch] = sc;
        }
      }
    }
    return states;
  }

  public toStateString(): string {
    const path = this.activePath;
    if (path.length === 0) return '';
    return path.map(n => {
      const scoreStr = n.scoreList.join('');
      return `${n.guess}.${scoreStr}`;
    }).join('.');
  }

  public loadStateString(stateStr: string) {
    const trimmed = stateStr.trim().toLowerCase();
    if (!trimmed) {
      this.reset();
      return;
    }

    const parts = trimmed.split('.');
    if (parts.length % 2 !== 0) {
      throw new Error(`Invalid state string: expected word.score pairs, got ${parts.length} tokens.`);
    }

    this.tree.reset();
    for (let i = 0; i < parts.length; i += 2) {
      const word = parts[i];
      const scoreNum = parseScoreString(parts[i + 1], this.wordList.wordLength);
      this.tree.addMove(word, scoreNum, this.wordList, this.activeStrategy, this.isWeighted);
    }
    this.notify();
  }
}

export function validateStateString(
  stateStr: string,
  wordList: WordList
): { isValid: boolean; candidateCount: number; isInconsistent: boolean; error?: string } {
  const trimmed = stateStr.trim().toLowerCase();
  if (!trimmed) {
    return { isValid: true, candidateCount: wordList.target.length, isInconsistent: false };
  }

  const parts = trimmed.split('.');
  if (parts.length % 2 !== 0) {
    return { isValid: false, candidateCount: 0, isInconsistent: false, error: 'Expected word.score pairs' };
  }

  let pool = [...wordList.target];
  for (let i = 0; i < parts.length; i += 2) {
    const word = parts[i];
    const scoreStr = parts[i + 1];

    if (word.length !== wordList.wordLength) {
      return { isValid: false, candidateCount: 0, isInconsistent: false, error: `"${word}" is not ${wordList.wordLength} letters` };
    }
    if (!wordList.allGuesses.includes(word)) {
      return { isValid: false, candidateCount: 0, isInconsistent: false, error: `"${word}" is not in dictionary` };
    }

    let scoreNum: number;
    try {
      scoreNum = parseScoreString(scoreStr, wordList.wordLength);
    } catch {
      return { isValid: false, candidateCount: 0, isInconsistent: false, error: `Invalid score "${scoreStr}"` };
    }

    pool = pool.filter(t => getScore(word, t) === scoreNum);
  }

  return {
    isValid: true,
    candidateCount: pool.length,
    isInconsistent: parts.length > 0 && pool.length === 0,
  };
}
