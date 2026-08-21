import { describe, it, expect, beforeAll } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import { getScore, parseScoreString, formatScoreEmoji } from './scoring';
import { parseWordListText, type WordList } from './wordlists';
import { GameEngine, validateStateString } from './state';
import { DecisionTreeStrategy } from './strategies';

describe('Wordlist & Strategy Parity', () => {
  let wordList: WordList;

  beforeAll(() => {
    const wordsPath = path.resolve(process.cwd(), 'public/data/words.txt');
    const text = fs.readFileSync(wordsPath, 'utf-8');
    wordList = parseWordListText(text);
  });

  it('loads exact 3,209 targets and 14,855 total guesses', () => {
    expect(wordList.target.length).toBe(3209);
    expect(wordList.allGuesses.length).toBe(14855);
  });

  it('accurately computes getScore for known cases', () => {
    // Exact match = 242 (22222 base 3 = 2*81 + 2*27 + 2*9 + 2*3 + 2 = 242)
    expect(getScore('pilot', 'pilot')).toBe(242);
    // tarse vs donut: T is yellow in donut (position 0 vs position 4) -> 10000 (81)
    expect(getScore('tarse', 'donut')).toBe(parseScoreString('10000'));
    // Double letters: guess 'bleed', target 'speed'
    expect(formatScoreEmoji(getScore('bleed', 'speed'))).toBe('⬛⬛🟩🟩🟩');
  });

  it('Decision Tree Opener matches C solver TARSE', () => {
    const treePath = path.resolve(process.cwd(), 'public/data/optimal_tree.json');
    const treeData = JSON.parse(fs.readFileSync(treePath, 'utf-8'));
    const treeStrategy = new DecisionTreeStrategy(treeData, wordList.target);
    const engine = new GameEngine(wordList, treeStrategy);

    const initialOpener = engine.getSuggestedNextGuess();
    expect(initialOpener?.guess).toBe('tarse');
    expect(initialOpener?.entropy).toBeCloseTo(5.8951, 2);
  });

  it('MoveTree supports branching and tree navigation', () => {
    const treePath = path.resolve(process.cwd(), 'public/data/optimal_tree.json');
    const treeData = JSON.parse(fs.readFileSync(treePath, 'utf-8'));
    const treeStrategy = new DecisionTreeStrategy(treeData, wordList.target);
    const engine = new GameEngine(wordList, treeStrategy);

    engine.setSecretSolution('pilot');
    const node1 = engine.addGuess('crane')!;
    expect(node1.guess).toBe('crane');
    expect(engine.activePath.length).toBe(1);

    // Step back to root
    engine.stepBack();
    expect(engine.activePath.length).toBe(0);

    // Add branch variation from root
    const node2 = engine.addGuess('tarse')!;
    expect(node2.guess).toBe('tarse');
    expect(engine.activePath.length).toBe(1);

    // Promote crane line back to main
    engine.promoteToMainLine(node1.id);
    engine.selectNode(node1.id);
    expect(engine.activePath[0].guess).toBe('crane');
  });

  it('detects inconsistent states correctly when 0 words match', () => {
    const engine = new GameEngine(wordList);
    engine.setMode('unknown');
    engine.addGuess('tarse', 242);
    engine.addGuess('guild', 242);
    expect(engine.isInconsistent).toBe(true);
    expect(engine.currentCandidates.length).toBe(0);
  });

  it('playOptimalMove works even from off-tree positions (falls back to entropy optimal)', () => {
    const treePath = path.resolve(process.cwd(), 'public/data/optimal_tree.json');
    const treeData = JSON.parse(fs.readFileSync(treePath, 'utf-8'));
    const treeStrategy = new DecisionTreeStrategy(treeData, wordList.target);
    const engine = new GameEngine(wordList, treeStrategy);

    engine.setSecretSolution('pilot');
    engine.addGuess('tarse');
    const resinNode = engine.addGuess('resin')!;
    expect(engine.activePath.length).toBe(2);

    // Select resinNode and play optimal move from off-tree state
    engine.selectNode(resinNode.id);
    const nextMove = engine.playOptimalMove();
    expect(nextMove).not.toBeNull();
    expect(nextMove?.guess.length).toBe(5);
    expect(engine.activePath.length).toBe(3);
  });

  it('validates state strings and reports candidate counts in real time', () => {
    const valid = validateStateString('tarse.10000.donut.01002', wordList);
    expect(valid.isValid).toBe(true);
    expect(valid.candidateCount).toBeGreaterThan(0);

    const invalid = validateStateString('invalid.10000', wordList);
    expect(invalid.isValid).toBe(false);
  });

  it('reproduces exact user tree: TARSE -> (CHALK -> SPADE, LAMPS, LOATH) and TARSE -> RESIN -> LOATH', () => {
    const treePath = path.resolve(process.cwd(), 'public/data/optimal_tree.json');
    const treeData = JSON.parse(fs.readFileSync(treePath, 'utf-8'));
    const treeStrategy = new DecisionTreeStrategy(treeData, wordList.target);
    const engine = new GameEngine(wordList, treeStrategy);

    // Turn 1
    const tarseNode = engine.addGuess('tarse')!;

    // Var 1 under TARSE: CHALK
    engine.selectNode(tarseNode.id);
    const chalkNode = engine.addGuess('chalk')!;

    // Sub-variations under CHALK
    engine.selectNode(chalkNode.id);
    engine.addGuess('spade');

    engine.selectNode(chalkNode.id);
    engine.addGuess('lamps');

    engine.selectNode(chalkNode.id);
    const loathVarNode = engine.addGuess('loath')!;

    // Main line under TARSE: RESIN
    engine.selectNode(tarseNode.id);
    const resinNode = engine.addGuess('resin')!;

    // Move under RESIN: LOATH
    engine.selectNode(resinNode.id);
    const loathMainNode = engine.addGuess('loath')!;

    // Test optimal move from loathVarNode
    engine.selectNode(loathVarNode.id);
    const nextVarMove = engine.playOptimalMove();
    expect(nextVarMove).not.toBeNull();

    // Test optimal move from loathMainNode
    engine.selectNode(loathMainNode.id);
    const nextMainMove = engine.playOptimalMove();
    expect(nextMainMove).not.toBeNull();
  });

  it('prevents playing the same word twice along the same branch', () => {
    const engine = new GameEngine(wordList);
    engine.setSecretSolution('pilot');
    const move1 = engine.addGuess('tarse');
    expect(move1).not.toBeNull();

    // Try to guess tarse again on the same active branch -> must return null
    const duplicateMove = engine.addGuess('tarse');
    expect(duplicateMove).toBeNull();
    expect(engine.activePath.length).toBe(1);

    // Playing a different word should work
    const move2 = engine.addGuess('donut');
    expect(move2).not.toBeNull();
    expect(engine.activePath.length).toBe(2);

    // Try guessing donut again -> must return null
    expect(engine.addGuess('donut')).toBeNull();
  });

  it('enforces strict 6-move maximum limit per game line', () => {
    const engine = new GameEngine(wordList);
    engine.setSecretSolution('pilot');
    const moves = ['tarse', 'crane', 'slate', 'roate', 'adieu', 'media'];
    for (const m of moves) {
      const node = engine.addGuess(m);
      expect(node).not.toBeNull();
    }
    expect(engine.activePath.length).toBe(6);

    // 7th move must be blocked
    const move7 = engine.addGuess('chalk');
    expect(move7).toBeNull();
    expect(engine.activePath.length).toBe(6);
  });
});
