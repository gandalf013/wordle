import { getScore, getScoreList, Score } from './scoring';
import type { WordList } from './wordlists';
import { analyze, analyzeAll } from './analysis';
import type { GuessAnalysis } from './analysis';
import { calculateSkill, calculateLuck } from './skill';
import { DecisionTreeStrategy, EntropyStrategy } from './strategies';
import type { Strategy } from './strategies';
import { TOP_OPENERS } from './top_openers';

export interface MoveNode {
  id: string;
  parentId: string | null;
  turnNumber: number;
  guess: string;
  score: number;
  scoreList: Score[];
  candidatesBefore: string[];
  candidatesAfter: string[];
  userAnalysis: GuessAnalysis;
  skill: number;
  luck: number;
  optimalGuess: string;
  optimalAnalysis: GuessAnalysis;
  isOptimal: boolean;
  children: MoveNode[];
}

export class MoveTree {
  public rootCandidates: string[];
  public nodes: Map<string, MoveNode> = new Map();
  public rootChildIds: string[] = [];
  public activeNodeId: string | null = null;
  private idCounter: number = 0;

  constructor(initialCandidates: string[]) {
    this.rootCandidates = initialCandidates;
  }

  public get activeNode(): MoveNode | null {
    return this.activeNodeId ? this.nodes.get(this.activeNodeId) || null : null;
  }

  public getActivePath(): MoveNode[] {
    const path: MoveNode[] = [];
    let curr = this.activeNode;
    while (curr) {
      path.unshift(curr);
      curr = curr.parentId ? this.nodes.get(curr.parentId) || null : null;
    }
    return path;
  }

  public getActiveCandidates(): string[] {
    const active = this.activeNode;
    return active ? active.candidatesAfter : this.rootCandidates;
  }

  public addMove(
    guess: string,
    score: number,
    wordList: WordList,
    strategy: Strategy,
    isWeighted: boolean = false
  ): MoveNode {
    const cleanGuess = guess.trim().toLowerCase();
    const parent = this.activeNode;
    const parentId = parent ? parent.id : null;
    const turnNumber = parent ? parent.turnNumber + 1 : 1;
    const candidatesBefore = parent ? parent.candidatesAfter : this.rootCandidates;

    // Check if an identical child already exists
    const siblings = parent ? parent.children : this.getRootChildren();
    const existing = siblings.find(c => c.guess === cleanGuess && c.score === score);
    if (existing) {
      this.activeNodeId = existing.id;
      return existing;
    }

    const weights = isWeighted ? wordList.weights : undefined;
    const userAnalysis = analyze(cleanGuess, candidatesBefore, weights, true);

    // Determine optimal guess
    let optimalGuess = cleanGuess;
    if (strategy instanceof DecisionTreeStrategy) {
      if (candidatesBefore.length === wordList.target.length) {
        optimalGuess = strategy.getOpener();
      } else {
        const treeOpt = strategy.findOptimalGuess(candidatesBefore);
        if (treeOpt) {
          optimalGuess = treeOpt;
        } else {
          const guessesToEvaluate = candidatesBefore.length <= 500 ? wordList.allGuesses : TOP_OPENERS;
          const analyses = analyzeAll(guessesToEvaluate, candidatesBefore, weights, false, true);
          const ranked = new EntropyStrategy().rank(analyses, isWeighted);
          if (ranked.length > 0) optimalGuess = ranked[0].guess;
        }
      }
    } else {
      const guessesToEvaluate = candidatesBefore.length <= 500 ? wordList.allGuesses : TOP_OPENERS;
      const analyses = analyzeAll(guessesToEvaluate, candidatesBefore, weights, false, true);
      const ranked = strategy.rank(analyses, isWeighted);
      if (ranked.length > 0) optimalGuess = ranked[0].guess;
    }
    const optimalAnalysis = analyze(optimalGuess, candidatesBefore, weights, true);
    const skill = calculateSkill(userAnalysis, [userAnalysis, optimalAnalysis]);
    const luck = calculateLuck(userAnalysis, score);

    // Filter candidates after
    const candidatesAfter = candidatesBefore.filter(t => getScore(cleanGuess, t) === score);

    const nodeId = `node_${++this.idCounter}_${cleanGuess}`;
    const newNode: MoveNode = {
      id: nodeId,
      parentId,
      turnNumber,
      guess: cleanGuess,
      score,
      scoreList: getScoreList(score, wordList.wordLength),
      candidatesBefore,
      candidatesAfter,
      userAnalysis,
      skill,
      luck,
      optimalGuess,
      optimalAnalysis,
      isOptimal: cleanGuess === optimalGuess,
      children: [],
    };

    this.nodes.set(nodeId, newNode);
    if (parent) {
      parent.children.push(newNode);
    } else {
      this.rootChildIds.push(nodeId);
    }

    this.activeNodeId = nodeId;
    return newNode;
  }

  public getRootChildren(): MoveNode[] {
    return this.rootChildIds
      .map(id => this.nodes.get(id))
      .filter((n): n is MoveNode => n !== undefined);
  }

  public selectNode(nodeId: string | null) {
    if (nodeId === null) {
      this.activeNodeId = null;
      return;
    }
    if (this.nodes.has(nodeId)) {
      this.activeNodeId = nodeId;
    }
  }

  public stepBack(): boolean {
    const curr = this.activeNode;
    if (!curr) return false;
    this.activeNodeId = curr.parentId;
    return true;
  }

  public stepForward(): boolean {
    const curr = this.activeNode;
    const children = curr ? curr.children : this.getRootChildren();
    if (children.length > 0) {
      this.activeNodeId = children[0].id;
      return true;
    }
    return false;
  }

  public promoteToMainLine(nodeId: string) {
    const node = this.nodes.get(nodeId);
    if (!node) return;

    if (!node.parentId) {
      // Root level child
      const idx = this.rootChildIds.indexOf(nodeId);
      if (idx > 0) {
        this.rootChildIds.splice(idx, 1);
        this.rootChildIds.unshift(nodeId);
      }
      return;
    }

    // Climb up tree promoting to index 0
    let curr: MoveNode | undefined = node;
    while (curr && curr.parentId) {
      const parent = this.nodes.get(curr.parentId);
      if (!parent) break;
      const idx = parent.children.findIndex(c => c.id === curr!.id);
      if (idx > 0) {
        const [promoted] = parent.children.splice(idx, 1);
        parent.children.unshift(promoted);
      }
      curr = parent;
    }
  }

  public deleteNode(nodeId: string) {
    const node = this.nodes.get(nodeId);
    if (!node) return;

    // Remove from parent
    if (node.parentId) {
      const parent = this.nodes.get(node.parentId);
      if (parent) {
        parent.children = parent.children.filter(c => c.id !== nodeId);
      }
    } else {
      this.rootChildIds = this.rootChildIds.filter(id => id !== nodeId);
    }

    // If active node was in deleted subtree, set active to parent
    const activePath = this.getActivePath();
    if (activePath.some(n => n.id === nodeId)) {
      this.activeNodeId = node.parentId;
    }

    // Recursively delete subtree nodes
    const deleteSubtree = (n: MoveNode) => {
      for (const child of n.children) deleteSubtree(child);
      this.nodes.delete(n.id);
    };
    deleteSubtree(node);
  }

  public updateNodeScore(
    nodeId: string,
    newScore: number,
    wordList: WordList,
    strategy: Strategy,
    isWeighted: boolean = false
  ) {
    const node = this.nodes.get(nodeId);
    if (!node) return;

    node.score = newScore;
    node.scoreList = getScoreList(newScore, wordList.wordLength);
    node.candidatesAfter = node.candidatesBefore.filter(t => getScore(node.guess, t) === newScore);
    node.luck = calculateLuck(node.userAnalysis, newScore);

    // Recompute children recursively
    const recomputeSubtree = (parent: MoveNode) => {
      const weights = isWeighted ? wordList.weights : undefined;
      for (const child of parent.children) {
        child.candidatesBefore = parent.candidatesAfter;
        child.userAnalysis = analyze(child.guess, child.candidatesBefore, weights, true);
        
        let optimalGuess = child.guess;
        if (strategy instanceof DecisionTreeStrategy) {
          const treeOpt = strategy.findOptimalGuess(child.candidatesBefore);
          if (treeOpt) {
            optimalGuess = treeOpt;
          } else {
            const guessesToEvaluate = child.candidatesBefore.length <= 500 ? wordList.allGuesses : TOP_OPENERS;
            const analyses = analyzeAll(guessesToEvaluate, child.candidatesBefore, weights, false, true);
            const ranked = new EntropyStrategy().rank(analyses, isWeighted);
            if (ranked.length > 0) optimalGuess = ranked[0].guess;
          }
        } else {
          const guessesToEvaluate = child.candidatesBefore.length <= 500 ? wordList.allGuesses : TOP_OPENERS;
          const analyses = analyzeAll(guessesToEvaluate, child.candidatesBefore, weights, false, true);
          const ranked = strategy.rank(analyses, isWeighted);
          if (ranked.length > 0) optimalGuess = ranked[0].guess;
        }
        child.optimalGuess = optimalGuess;
        child.optimalAnalysis = analyze(optimalGuess, child.candidatesBefore, weights, true);
        child.isOptimal = child.guess === optimalGuess;
        child.skill = calculateSkill(child.userAnalysis, [child.userAnalysis, child.optimalAnalysis]);
        child.candidatesAfter = child.candidatesBefore.filter(t => getScore(child.guess, t) === child.score);
        child.luck = calculateLuck(child.userAnalysis, child.score);
        recomputeSubtree(child);
      }
    };
    recomputeSubtree(node);
  }

  public recomputeAll(
    wordList: WordList,
    strategy: Strategy,
    isWeighted: boolean = false
  ) {
    const weights = isWeighted ? wordList.weights : undefined;
    const recompute = (node: MoveNode) => {
      node.userAnalysis = analyze(node.guess, node.candidatesBefore, weights, true);
      let optimalGuess = node.guess;
      if (strategy instanceof DecisionTreeStrategy) {
        if (node.candidatesBefore.length === wordList.target.length) {
          optimalGuess = strategy.getOpener();
        } else {
          const treeOpt = strategy.findOptimalGuess(node.candidatesBefore, wordList.allGuesses);
          if (treeOpt) optimalGuess = treeOpt;
        }
      } else {
        const guessesToEvaluate = node.candidatesBefore.length <= 500 ? wordList.allGuesses : TOP_OPENERS;
        const analyses = analyzeAll(guessesToEvaluate, node.candidatesBefore, weights, false, true);
        const ranked = strategy.rank(analyses, isWeighted);
        if (ranked.length > 0) optimalGuess = ranked[0].guess;
      }
      node.optimalGuess = optimalGuess;
      node.optimalAnalysis = analyze(optimalGuess, node.candidatesBefore, weights, true);
      node.isOptimal = node.guess === optimalGuess;
      node.skill = calculateSkill(node.userAnalysis, [node.userAnalysis, node.optimalAnalysis]);
      node.luck = calculateLuck(node.userAnalysis, node.score);

      for (const child of node.children) {
        recompute(child);
      }
    };

    for (const root of this.getRootChildren()) {
      recompute(root);
    }
  }

  public reset() {
    this.nodes.clear();
    this.rootChildIds = [];
    this.activeNodeId = null;
    this.idCounter = 0;
  }
}
