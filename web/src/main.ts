import './style.css';
import { loadWordList } from './engine/wordlists';
import type { WordList } from './engine/wordlists';
import { loadDecisionTree, ALL_STRATEGIES, DecisionTreeStrategy, TwoPlyExpectimaxStrategy } from './engine/strategies';
import { GameEngine, validateStateString, type MoveNode } from './engine/state';
import { formatScoreEmoji, Score } from './engine/scoring';
import { analyze, analyzeAll } from './engine/analysis';
import type { GuessAnalysis } from './engine/analysis';
import { TOP_OPENERS } from './engine/top_openers';
import { APP_CONFIG } from './config';

let wordList: WordList;
let engine: GameEngine;
let currentTypingWord = '';
let activeRightTab: 'candidates' | 'topn' | 'bins' = 'candidates';
let candidateSearchQuery = '';
let topNLimit = 15;
let peekWordQuery = '';
let peekWordAnalysis: GuessAnalysis | null = null;
let expandedBucketScore: number | null = null;
let isComputingOptimal = false;
let isAutoSolving = false;
let autoSolveAbort = false;
let treeViewMode: 'inline' | 'tree' = 'inline';
let toastTimeout: any = null;

export function showToast(message: string, type: 'info' | 'success' | 'warning' = 'info', duration: number = 3000) {
  let toastContainer = document.getElementById('toast-container');
  if (!toastContainer) {
    toastContainer = document.createElement('div');
    toastContainer.id = 'toast-container';
    document.body.appendChild(toastContainer);
  }

  const icon = type === 'success' ? '✅' : type === 'warning' ? '⚠️' : 'ℹ️';
  toastContainer.innerHTML = `
    <div class="toast-pill toast-${type}">
      <span class="toast-icon">${icon}</span>
      <span class="toast-text">${message}</span>
    </div>
  `;

  if (toastTimeout) clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => {
    if (toastContainer) toastContainer.innerHTML = '';
  }, duration);
}

async function init() {
  const app = document.getElementById('app')!;
  app.innerHTML = `
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 70vh; gap: 16px;">
      <div class="spinner"></div>
      <div style="font-family: var(--font-mono); font-size: 14px; color: var(--text-secondary);">
        Loading 14,855 dictionary words & optimal decision tree...
      </div>
    </div>
  `;

  try {
    wordList = await loadWordList();
    const treeStrategy = await loadDecisionTree(wordList.target);
    engine = new GameEngine(wordList, treeStrategy);

    // Subscribe to engine state updates for reactive re-render
    engine.subscribe(() => {
      render();
    });

    // Check URL parameters for initial game state
    const params = new URLSearchParams(window.location.search);
    const targetParam = params.get('target');
    const stateParam = params.get('state') || params.get('moves');
    if (targetParam && targetParam.length === 5) {
      engine.setSecretSolution(targetParam);
      engine.setMode('known');
    }
    if (stateParam) {
      try {
        engine.loadStateString(stateParam);
      } catch (e) {
        console.warn('Could not load initial state from URL:', e);
      }
    }

    render();
    window.addEventListener('keydown', handlePhysicalKeyboard);
  } catch (err) {
    app.innerHTML = `
      <div style="padding: 30px; color: var(--accent-red); font-family: var(--font-mono);">
        <h3>Error initializing Wordle Explorer</h3>
        <p>${(err as Error).message}</p>
      </div>
    `;
  }
}

function handlePhysicalKeyboard(e: KeyboardEvent) {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement || e.target instanceof HTMLTextAreaElement) return;

  if (e.key === 'ArrowLeft') {
    e.preventDefault();
    engine.stepBack();
  } else if (e.key === 'ArrowRight') {
    e.preventDefault();
    engine.stepForward();
  } else if (e.key === 'Backspace') {
    handleKeyPress('BACKSPACE');
  } else if (e.key === 'Enter') {
    handleKeyPress('ENTER');
  } else if (/^[a-zA-Z]$/.test(e.key)) {
    handleKeyPress(e.key.toUpperCase());
  }
}

function updateTypingRow() {
  const activePath = engine.activePath;
  const rowIndex = activePath.length;
  if (rowIndex >= 6) return;
  const rowEl = document.querySelector(`.grid-row[data-row="${rowIndex}"]`);
  if (!rowEl) {
    render();
    return;
  }
  const tiles = rowEl.querySelectorAll('.grid-tile');
  for (let c = 0; c < 5; c++) {
    const tile = tiles[c] as HTMLElement;
    if (!tile) continue;
    const letter = currentTypingWord[c] || '';
    tile.textContent = letter;
    tile.className = letter ? 'grid-tile filled' : 'grid-tile';
  }
}

function handleKeyPress(key: string) {
  if (key === 'BACKSPACE') {
    if (currentTypingWord.length > 0) {
      currentTypingWord = currentTypingWord.slice(0, -1);
      updateTypingRow();
    }
  } else if (key === 'ENTER') {
    if (currentTypingWord.length === 5) {
      if (wordList.allGuesses.includes(currentTypingWord.toLowerCase())) {
        const guessToPlay = currentTypingWord;
        currentTypingWord = '';
        engine.addGuess(guessToPlay);
      } else {
        showToast(`"${currentTypingWord}" is not in the dictionary`, 'warning');
      }
    }
  } else if (/^[A-Z]$/.test(key)) {
    if (currentTypingWord.length < 5 && engine.activePath.length < 6) {
      currentTypingWord += key;
      updateTypingRow();
    }
  }
}

function render() {
  const app = document.getElementById('app')!;
  const keyStates = engine.getKeyboardKeyStates();
  const activePath = engine.activePath;
  const activeNode = engine.tree.activeNode;
  const candidates = engine.currentCandidates;

  // Candidate match badge
  let matchBadgeHtml = '';
  if (engine.isInconsistent) {
    matchBadgeHtml = `
      <div class="candidates-pill error">
        ⚠️ 0 words match (Inconsistent clues)
      </div>
    `;
  } else if (candidates.length === 1) {
    matchBadgeHtml = `
      <div class="candidates-pill solved">
        🎯 1 word matches: <strong>${candidates[0].toUpperCase()}</strong>
      </div>
    `;
  } else {
    matchBadgeHtml = `
      <div class="candidates-pill normal">
        🟢 <strong>${candidates.length.toLocaleString()}</strong> words match
      </div>
    `;
  }

  app.innerHTML = `
    <!-- HEADER -->
    <header class="app-header">
      <div class="header-left">
        <span class="app-logo">🟩🟨⬛</span>
        <div>
          <h1 class="app-title">${APP_CONFIG.name}</h1>
          <span class="app-subtitle">Lichess-style Analysis & Optimal Solver Studio</span>
        </div>
      </div>

      <!-- Strategy & Mode Header Controls -->
      <div class="header-center">
        <!-- Mode Switcher -->
        <div class="mode-toggle-group">
          <button
            class="mode-btn ${engine.mode === 'known' ? 'active' : ''}"
            data-mode="known"
            title="Auto-score moves against a specified or random secret word"
          >
            🎯 Known Word
          </button>
          <button
            class="mode-btn ${engine.mode === 'unknown' ? 'active' : ''}"
            data-mode="unknown"
            title="Manual tile coloring sandbox for daily NYT / custom puzzles"
          >
            ❓ Sandbox
          </button>
        </div>

        <!-- Strategy Selector (First-Class Control) -->
        <div class="strategy-header-group">
          <span class="bar-label">STRATEGY:</span>
          <select id="global-strategy-select" class="select-styled strategy-select-main">
            ${Object.values(ALL_STRATEGIES).map(s => `
              <option value="${s.name}" ${s.name === engine.activeStrategy.name ? 'selected' : ''}>
                ${s.label}
              </option>
            `).join('')}
          </select>
          <label class="toggle-control-label" title="Weight word frequencies by usage">
            <input type="checkbox" id="global-weighted-toggle" ${engine.isWeighted ? 'checked' : ''} />
            <span>Weighted</span>
          </label>
        </div>
      </div>

      <div class="header-actions">
        <button id="btn-theme-toggle" class="btn-icon" title="Toggle Dark/Light Theme">🌓</button>
        <button id="btn-share-url" class="btn-action" style="padding: 6px 12px; font-size: 12px;" title="Copy share link">🔗 Share</button>
      </div>
    </header>

    <!-- MODE CONTROLS BAR -->
    <div class="mode-bar">
      ${engine.mode === 'known' ? `
        <div class="known-word-bar">
          <span class="bar-label">TARGET:</span>
          <div class="target-input-wrapper">
            <input
              id="secret-target-input"
              class="secret-input ${engine.hideSecret ? 'masked' : ''}"
              type="text"
              maxlength="5"
              value="${engine.secretSolution ? (engine.hideSecret ? '•••••' : engine.secretSolution.toUpperCase()) : ''}"
              placeholder="e.g. PILOT"
              autocomplete="off"
              spellcheck="false"
            />
          </div>
          <button id="btn-toggle-mask" class="btn-action" style="padding: 6px 10px; font-size: 12px;" title="${engine.hideSecret ? 'Reveal target' : 'Hide target'}">
            ${engine.hideSecret ? '👁️ Reveal' : '🔒 Hide'}
          </button>
          <button id="btn-random-secret" class="btn-action" style="padding: 6px 12px; font-size: 12px;" title="Pick random target">
            🎲 Random
          </button>
          <button id="btn-today-wordle" class="btn-action" style="padding: 6px 12px; font-size: 12px;" title="Load Today's NYT Wordle word">
            📅 Today's Wordle
          </button>
        </div>
      ` : `
        <div class="unknown-word-bar">
          <span class="bar-label">STATE:</span>
          <input
            id="state-input-box"
            class="state-input"
            type="text"
            placeholder="e.g. tarse.10000.donut.01002"
            value="${engine.toStateString()}"
            spellcheck="false"
          />
          <button id="btn-load-state" class="btn-action" style="padding: 6px 12px; font-size: 12px;">Load State</button>
          <span id="state-feedback-pill" class="state-feedback-tag valid">
            🟢 ${candidates.length} candidates
          </span>
        </div>
      `}
    </div>

    <!-- MAIN TWO-COLUMN STUDIO LAYOUT -->
    <main class="studio-layout">
      <!-- LEFT PANEL: WORDLE BOARD & TOOLBAR -->
      <section class="left-panel">
        <div class="board-top-meta">
          ${matchBadgeHtml}
        </div>

        <!-- 6x5 GRID -->
        <div class="grid-container">
          ${renderBoard(activePath)}
        </div>

        <!-- TOOLBAR (Lichess Analysis Toolbar) -->
        <div class="analysis-toolbar">
          <button
            id="btn-play-optimal"
            class="btn-primary-action ${isComputingOptimal ? 'computing' : ''}"
            ${isComputingOptimal || isAutoSolving || activePath.length >= 6 || (activeNode && activeNode.score === 242) ? 'disabled' : ''}
            title="Play the single next optimal move (${engine.activeStrategy.label})"
          >
            ${isComputingOptimal ? '⏳ Thinking...' : ((activeNode && activeNode.score === 242) ? '🎉 Solved' : (activePath.length >= 6 ? '🛑 Max 6 Moves' : '🤖 Play Next Optimal'))}
          </button>
          <button
            id="btn-auto-solve"
            class="btn-action btn-auto-solve ${isAutoSolving ? 'active-solving' : ''}"
            ${(!isAutoSolving && (activePath.length >= 6 || (activeNode && activeNode.score === 242))) ? 'disabled' : ''}
            title="Automatically solve the game using the selected strategy"
          >
            ${isAutoSolving ? '⏹ Stop Solving' : '⚡ Auto-Solve'}
          </button>
          <button id="btn-toggle-show-optimal" class="btn-action ${engine.showOptimalCard ? 'active' : ''}" title="Toggle optimal move comparison card" ${isComputingOptimal || isAutoSolving ? 'disabled' : ''}>
            💡 ${engine.showOptimalCard ? 'Hide Optimal' : 'Show Optimal'}
          </button>
          <button id="btn-step-back" class="btn-icon-action" title="Step back (Left Arrow)" ${activePath.length === 0 || isComputingOptimal || isAutoSolving ? 'disabled' : ''}>
            ⎌ Step Back
          </button>
          <button id="btn-reset-game" class="btn-icon-action" title="Start fresh / new game" ${isComputingOptimal || isAutoSolving ? 'disabled' : ''}>
            ⟲ Reset
          </button>
        </div>

        <!-- VIRTUAL KEYBOARD -->
        <div class="keyboard-container">
          ${renderKeyboard(keyStates)}
        </div>
      </section>

      <!-- RIGHT PANEL: LICHESS-STYLE ANALYSIS STUDIO -->
      <section class="right-panel">
        <!-- 1. LICHESS MOVE TREE & VARIATIONS -->
        <div class="panel-card move-tree-card">
          <div class="card-header" style="display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 8px;">
              <span class="card-title">ANALYSIS MOVE TREE</span>
              <span style="font-size: 11px; color: var(--text-muted);">(Click any move to jump)</span>
            </div>
            <div class="tree-view-toggle">
              <button class="btn-toggle-view ${treeViewMode === 'inline' ? 'active' : ''}" data-tree-view="inline" title="Inline notation (PGN-style)">
                📄 Inline
              </button>
              <button class="btn-toggle-view ${treeViewMode === 'tree' ? 'active' : ''}" data-tree-view="tree" title="Vertical branching tree graph">
                🌳 Tree Graph
              </button>
            </div>
          </div>
          <div class="move-tree-body">
            ${renderMoveTree(engine.tree)}
          </div>
        </div>

        <!-- 2. TURN METRICS & OPTIMAL COMPARISON CARD -->
        ${activeNode ? renderTurnMetrics(activeNode) : ''}

        <!-- 3. DEEP-DIVE TABS: REMAINING WORDS, TOP-N, BINS -->
        <div class="panel-card tabbed-analysis-card">
          <div class="tabs-header">
            <button class="tab-btn ${activeRightTab === 'candidates' ? 'active' : ''}" data-tab="candidates">
              🎯 Remaining Words (${candidates.length})
            </button>
            <button class="tab-btn ${activeRightTab === 'topn' ? 'active' : ''}" data-tab="topn">
              🏆 Top Guess Suggestions
            </button>
            <button class="tab-btn ${activeRightTab === 'bins' ? 'active' : ''}" data-tab="bins">
              📦 Partition Bins
            </button>
          </div>

          <div class="tab-content">
            ${renderTabContent()}
          </div>
        </div>
      </section>
    </main>
  `;

  attachEventListeners();
}

function renderBoard(activePath: MoveNode[]): string {
  let rowsHtml = '';
  for (let r = 0; r < 6; r++) {
    if (r < activePath.length) {
      const node = activePath[r];
      const isSelected = engine.tree.activeNode?.id === node.id;
      const isManual = engine.mode === 'unknown';
      const tilesHtml = node.guess.split('').map((ch, c) => {
        const sc = node.scoreList[c];
        const colorClass = sc === Score.GREEN ? 'green' : sc === Score.YELLOW ? 'yellow' : 'gray';
        return `
          <div
            class="grid-tile ${colorClass} ${isManual ? 'clickable-tile' : ''}"
            data-node-id="${node.id}"
            data-letter-index="${c}"
            title="${isManual ? 'Click to cycle color (Gray -> Yellow -> Green)' : ''}"
          >
            ${ch.toUpperCase()}
          </div>
        `;
      }).join('');

      rowsHtml += `
        <div class="grid-row completed ${isSelected ? 'active-row' : ''}" data-row="${r}">
          ${tilesHtml}
        </div>
      `;
    } else if (r === activePath.length) {
      const tilesHtml = Array.from({ length: 5 }).map((_, c) => {
        const letter = currentTypingWord[c] || '';
        return `
          <div class="grid-tile ${letter ? 'filled' : ''}">
            ${letter}
          </div>
        `;
      }).join('');

      rowsHtml += `
        <div class="grid-row typing" data-row="${r}">
          ${tilesHtml}
        </div>
      `;
    } else {
      const tilesHtml = Array.from({ length: 5 }).map(() => `
        <div class="grid-tile"></div>
      `).join('');

      rowsHtml += `
        <div class="grid-row empty" data-row="${r}">
          ${tilesHtml}
        </div>
      `;
    }
  }
  return rowsHtml;
}

function renderKeyboard(keyStates: Record<string, Score>): string {
  const rows = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    ['ENTER', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', 'BACKSPACE'],
  ];

  return rows.map(row => `
    <div class="keyboard-row">
      ${row.map(k => {
        const state = keyStates[k];
        let colorClass = '';
        if (state === Score.GREEN) colorClass = 'key-green';
        else if (state === Score.YELLOW) colorClass = 'key-yellow';
        else if (state === Score.GRAY) colorClass = 'key-gray';
        const isWide = k === 'ENTER' || k === 'BACKSPACE';
        const label = k === 'BACKSPACE' ? '⌫' : k;

        return `
          <button class="key-btn ${colorClass} ${isWide ? 'wide' : ''}" data-key="${k}">
            ${label}
          </button>
        `;
      }).join('')}
    </div>
  `).join('');
}

function renderInlineTree(tree: typeof engine.tree): string {
  const rootChildren = tree.getRootChildren();
  if (rootChildren.length === 0) {
    return `
      <div style="padding: 16px; color: var(--text-muted); font-size: 13px; text-align: center;">
        No moves played yet. Type a word or click <strong>Play Next Optimal</strong> to begin.
      </div>
    `;
  }

  const activeId = tree.activeNode?.id || null;

  function renderMove(n: MoveNode, isMain: boolean): string {
    const isActive = n.id === activeId;
    const emojiStr = formatScoreEmoji(n.score);
    const numPrefix = `${n.turnNumber}.`;

    return `
      <button
        class="lichess-move ${isMain ? 'main-line-move' : 'variation-move'} ${isActive ? 'active' : ''}"
        data-select-node="${n.id}"
        title="Turn ${n.turnNumber}: ${n.guess.toUpperCase()} (Skill: ${n.skill}/100, Luck: ${n.luck}/100)"
      >
        <span class="m-num">${numPrefix}</span>
        <span class="m-word">${n.guess.toUpperCase()}</span>
        <span class="m-emoji">${emojiStr}</span>
        <span class="m-pool">(${n.candidatesAfter.length})</span>
        ${n.isOptimal ? '<span class="m-star" title="Optimal move according to strategy">⭐</span>' : ''}
      </button>
    `;
  }

  function renderNodeAndContinuation(n: MoveNode, isMain: boolean): string {
    let out = renderMove(n, isMain);

    // If this node has alternate branching siblings (variations)
    if (n.children.length > 1) {
      const mainChild = n.children[0];
      const variations = n.children.slice(1);

      // Render variations in parentheses
      for (const v of variations) {
        out += `
          <span class="lichess-variation-inline">
            <span class="var-paren">(</span>
            ${renderVariationLine(v)}
            <span class="var-actions">
              <button class="btn-var-action" data-promote-id="${v.id}" title="Promote variation to Main Line">⤊</button>
              <button class="btn-var-action" data-delete-id="${v.id}" title="Delete variation">✕</button>
            </span>
            <span class="var-paren">)</span>
          </span>
        `;
      }

      // Continue main line child
      out += renderNodeAndContinuation(mainChild, isMain);
    } else if (n.children.length === 1) {
      out += renderNodeAndContinuation(n.children[0], isMain);
    }

    return out;
  }

  function renderVariationLine(v: MoveNode): string {
    let out = renderMove(v, false);
    if (v.children.length > 1) {
      const mainChild = v.children[0];
      const subVars = v.children.slice(1);
      for (const sv of subVars) {
        out += `
          <span class="lichess-variation-inline sub-var">
            <span class="var-paren">(</span>
            ${renderVariationLine(sv)}
            <span class="var-actions">
              <button class="btn-var-action" data-promote-id="${sv.id}" title="Promote to Parent Line">⤊</button>
              <button class="btn-var-action" data-delete-id="${sv.id}" title="Delete sub-variation">✕</button>
            </span>
            <span class="var-paren">)</span>
          </span>
        `;
      }
      out += renderVariationLine(mainChild);
    } else if (v.children.length === 1) {
      out += renderVariationLine(v.children[0]);
    }
    return out;
  }

  const mainRoot = rootChildren[0];
  const altRoots = rootChildren.slice(1);

  let html = `<div class="lichess-main-flow">${renderNodeAndContinuation(mainRoot, true)}</div>`;

  if (altRoots.length > 0) {
    html += `
      <div class="lichess-alt-openers">
        <span class="alt-label">Alternate Openings:</span>
        ${altRoots.map(alt => `
          <span class="lichess-variation-inline">
            <span class="var-paren">(</span>
            ${renderVariationLine(alt)}
            <span class="var-actions">
              <button class="btn-var-action" data-promote-id="${alt.id}" title="Promote Opener to Main Line">⤊</button>
              <button class="btn-var-action" data-delete-id="${alt.id}" title="Delete Opener">✕</button>
            </span>
            <span class="var-paren">)</span>
          </span>
        `).join('')}
      </div>
    `;
  }

  return `<div class="lichess-move-tree-container">${html}</div>`;
}

function renderVerticalTree(tree: typeof engine.tree): string {
  const rootChildren = tree.getRootChildren();
  if (rootChildren.length === 0) {
    return `
      <div style="padding: 16px; color: var(--text-muted); font-size: 13px; text-align: center;">
        No moves played yet. Type a word or click <strong>Play Next Optimal</strong> to begin.
      </div>
    `;
  }

  const activeId = tree.activeNode?.id || null;

  function renderNodeRow(n: MoveNode, isMain: boolean): string {
    const isActive = n.id === activeId;
    const emojiStr = formatScoreEmoji(n.score);

    return `
      <div class="tree-node-row ${isMain ? 'main-line-row' : 'variation-line-row'} ${isActive ? 'active-tree-node' : ''}">
        <button
          class="tree-node-pill ${isMain ? 'pill-main' : 'pill-var'} ${isActive ? 'active' : ''}"
          data-select-node="${n.id}"
          title="Turn ${n.turnNumber}: ${n.guess.toUpperCase()} (Skill: ${n.skill}/100, Luck: ${n.luck}/100)"
        >
          <span class="m-num">${n.turnNumber}.</span>
          <span class="m-word">${n.guess.toUpperCase()}</span>
          <span class="m-emoji">${emojiStr}</span>
          <span class="m-pool">(${n.candidatesAfter.length})</span>
          ${n.isOptimal ? '<span class="m-star" title="Strategy Optimal">⭐</span>' : ''}
        </button>
        <div class="tree-node-meta">
          <span class="meta-skill" title="Skill Score">🎯 ${n.skill}%</span>
          <span class="meta-luck" title="Luck Score">🎲 ${n.luck}%</span>
          ${!isMain ? `
            <button class="btn-var-action" data-promote-id="${n.id}" title="Promote to Main Line">⤊</button>
            <button class="btn-var-action" data-delete-id="${n.id}" title="Delete variation">✕</button>
          ` : `
            <button class="btn-var-action" data-delete-id="${n.id}" title="Delete node">✕</button>
          `}
        </div>
      </div>
    `;
  }

  function renderMainLineNodeAndVariations(n: MoveNode): string {
    let html = renderNodeRow(n, true);

    // If there are variations branching off this node (children 1, 2, ...)
    if (n.children.length > 1) {
      for (let i = 1; i < n.children.length; i++) {
        html += `
          <div class="tree-variation-branch-container">
            <div class="branch-connector">└─</div>
            <div class="variation-chain">
              ${renderVariationChain(n.children[i])}
            </div>
          </div>
        `;
      }
    }

    // Continue main line child (child 0) directly below on the main spine
    if (n.children.length > 0) {
      html += renderMainLineNodeAndVariations(n.children[0]);
    }

    return html;
  }

  function renderVariationChain(v: MoveNode): string {
    let html = renderNodeRow(v, false);

    // If this variation has sub-variations (children 1, 2, ...)
    if (v.children.length > 1) {
      for (let i = 1; i < v.children.length; i++) {
        html += `
          <div class="tree-variation-branch-container sub-branch">
            <div class="branch-connector">└─</div>
            <div class="variation-chain">
              ${renderVariationChain(v.children[i])}
            </div>
          </div>
        `;
      }
    }

    // Continue variation main child
    if (v.children.length > 0) {
      html += renderVariationChain(v.children[0]);
    }

    return html;
  }

  const mainRoot = rootChildren[0];
  const altRoots = rootChildren.slice(1);

  let html = `<div class="vertical-main-spine">${renderMainLineNodeAndVariations(mainRoot)}</div>`;

  if (altRoots.length > 0) {
    html += `
      <div class="vertical-alt-openers">
        <div class="alt-label">Alternate Openings:</div>
        ${altRoots.map(alt => `
          <div class="tree-variation-branch-container">
            <div class="branch-connector">└─</div>
            <div class="variation-chain">
              ${renderVariationChain(alt)}
            </div>
          </div>
        `).join('')}
      </div>
    `;
  }

  return `<div class="vertical-tree-graph">${html}</div>`;
}

function renderMoveTree(tree: typeof engine.tree): string {
  if (treeViewMode === 'tree') {
    return renderVerticalTree(tree);
  }
  return renderInlineTree(tree);
}

function renderTurnMetrics(node: MoveNode): string {
  const reductionPct = node.candidatesBefore.length > 0
    ? (((node.candidatesBefore.length - node.candidatesAfter.length) / node.candidatesBefore.length) * 100).toFixed(1)
    : '0';

  let skillBadge = '<span class="metric-badge green">Optimal</span>';
  if (node.skill < 50) skillBadge = '<span class="metric-badge red">Inaccurate</span>';
  else if (node.skill < 80) skillBadge = '<span class="metric-badge yellow">Good</span>';
  else if (node.skill < 95) skillBadge = '<span class="metric-badge blue">Excellent</span>';

  let luckBadge = '<span class="metric-badge blue">Average</span>';
  if (node.luck > 75) luckBadge = '<span class="metric-badge green">Lucky</span>';
  else if (node.luck < 25) luckBadge = '<span class="metric-badge red">Unlucky</span>';

  return `
    <div class="panel-card turn-metrics-card">
      <div class="card-header">
        <span class="card-title">TURN ${node.turnNumber} EVALUATION: <strong>${node.guess.toUpperCase()}</strong></span>
        <span style="font-size: 12px; color: var(--text-secondary);">${formatScoreEmoji(node.score)}</span>
      </div>

      <div class="metrics-grid">
        <div class="metric-box">
          <div class="metric-label">SKILL SCORE</div>
          <div class="metric-value-row">
            <span class="metric-num">${node.skill}</span>
            <span class="metric-denom">/100</span>
            ${skillBadge}
          </div>
        </div>

        <div class="metric-box">
          <div class="metric-label">LUCK SCORE</div>
          <div class="metric-value-row">
            <span class="metric-num">${node.luck}</span>
            <span class="metric-denom">/100</span>
            ${luckBadge}
          </div>
        </div>

        <div class="metric-box">
          <div class="metric-label">POOL REDUCTION</div>
          <div class="metric-value-row">
            <span class="metric-num">${node.candidatesBefore.length} ➔ ${node.candidatesAfter.length}</span>
            <span class="metric-denom" style="color: var(--accent-green);">(-${reductionPct}%)</span>
          </div>
        </div>
      </div>

      ${engine.showOptimalCard ? `
        <div class="optimal-comparison-box">
          <div class="comp-header">
            <span style="font-size: 11px; font-weight: 700; color: var(--accent-blue); text-transform: uppercase;">
              💡 Strategy Recommendation at Turn ${node.turnNumber}
            </span>
          </div>
          <div class="comp-body">
            <div class="comp-side">
              <span class="comp-label">YOUR MOVE:</span>
              <strong class="comp-word">${node.guess.toUpperCase()}</strong>
              <span class="comp-stat">Entropy: ${node.userAnalysis.entropy.toFixed(3)} bits</span>
              <span class="comp-stat">Exp. Left: ${node.userAnalysis.expectedSize.toFixed(1)} words</span>
            </div>
            <div class="comp-vs">VS</div>
            <div class="comp-side optimal-side">
              <span class="comp-label">BEST MOVE:</span>
              <strong class="comp-word">${node.optimalGuess.toUpperCase()}</strong>
              <span class="comp-stat">Entropy: ${node.optimalAnalysis.entropy.toFixed(3)} bits</span>
              <span class="comp-stat">Exp. Left: ${node.optimalAnalysis.expectedSize.toFixed(1)} words</span>
            </div>
          </div>
        </div>
      ` : ''}
    </div>
  `;
}

function renderTabContent(): string {
  if (activeRightTab === 'candidates') {
    return renderCandidatesTab();
  } else if (activeRightTab === 'topn') {
    return renderTopNTab();
  } else {
    return renderBinsTab();
  }
}

function renderCandidatesTab(): string {
  const pool = engine.currentCandidates;
  const filtered = candidateSearchQuery
    ? pool.filter(w => w.includes(candidateSearchQuery.toLowerCase()))
    : pool;

  return `
    <div style="margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between; gap: 12px;">
      <input
        id="candidate-search-input"
        class="state-input"
        style="padding: 6px 12px; font-size: 12px; max-width: 250px;"
        type="text"
        placeholder="🔍 Search candidate words..."
        value="${candidateSearchQuery}"
      />
      <span style="font-size: 12px; color: var(--text-muted);">
        Showing <strong>${filtered.length}</strong> of ${pool.length}
      </span>
    </div>

    <div class="candidate-chips-container">
      ${filtered.slice(0, 200).map(w => `
        <button class="candidate-chip" data-play-word="${w}" title="Click to play ${w.toUpperCase()}">
          ${w.toUpperCase()}
        </button>
      `).join('')}
      ${filtered.length > 200 ? `
        <span style="font-size: 11px; color: var(--text-muted); padding: 6px;">
          +${filtered.length - 200} more candidates
        </span>
      ` : ''}
      ${filtered.length === 0 ? `
        <div style="padding: 20px; color: var(--text-muted); font-size: 13px; text-align: center; width: 100%;">
          No matching candidates found for "${candidateSearchQuery}".
        </div>
      ` : ''}
    </div>
  `;
}

function renderTopNTab(): string {
  const pool = engine.currentCandidates;
  const weights = engine.isWeighted ? wordList.weights : undefined;

  let ranked: GuessAnalysis[] = [];
  let decisionTreeOptimal: string | null = null;

  const candidateGuesses = pool.length <= 500 ? wordList.allGuesses : TOP_OPENERS;
  const scoredAnalyses = analyzeAll(candidateGuesses, pool, weights, false, true);

  if (engine.activeStrategy instanceof DecisionTreeStrategy) {
    decisionTreeOptimal = (engine.activeStrategy as DecisionTreeStrategy).findOptimalGuess(pool, wordList.allGuesses);
    const heuristicRanked = new TwoPlyExpectimaxStrategy().rank(scoredAnalyses, engine.isWeighted);
    if (decisionTreeOptimal) {
      const optAnalysis = scoredAnalyses.find(a => a.guess === decisionTreeOptimal) || analyze(decisionTreeOptimal, pool, weights, false);
      ranked = [optAnalysis, ...heuristicRanked.filter((a: GuessAnalysis) => a.guess !== decisionTreeOptimal)];
    } else {
      ranked = heuristicRanked;
    }
  } else {
    ranked = engine.activeStrategy.rank(scoredAnalyses, engine.isWeighted);
  }

  const displayLimit = topNLimit === -1 ? ranked.length : topNLimit;

  return `
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 14px;">
      <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
        <span style="font-size: 12px; color: var(--text-secondary);">
          Strategy: <strong>${engine.activeStrategy.label}</strong> ${engine.isWeighted ? '<span class="metric-badge blue">Weighted</span>' : ''}
        </span>

        <select id="topn-limit-select" class="select-styled" style="padding: 4px 10px; font-size: 12px;">
          <option value="10" ${topNLimit === 10 ? 'selected' : ''}>Top 10</option>
          <option value="15" ${topNLimit === 15 ? 'selected' : ''}>Top 15</option>
          <option value="25" ${topNLimit === 25 ? 'selected' : ''}>Top 25</option>
          <option value="50" ${topNLimit === 50 ? 'selected' : ''}>Top 50</option>
          <option value="-1" ${topNLimit === -1 ? 'selected' : ''}>Show All (${ranked.length})</option>
        </select>
      </div>

      <div style="display: flex; align-items: center; gap: 6px;">
        <input
          id="peek-word-input"
          class="state-input"
          style="padding: 4px 8px; font-size: 11px; width: 130px;"
          placeholder="Peek any word..."
          value="${peekWordQuery}"
        />
        <button id="btn-peek-search" class="btn-action" style="padding: 4px 8px; font-size: 11px;">🔍 Peek</button>
      </div>
    </div>

    ${peekWordAnalysis ? `
      <div style="background: var(--bg-surface-raised); border: 1px solid var(--accent-blue); border-radius: var(--radius-md); padding: 10px 14px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;">
        <div>
          <strong style="font-size: 16px; font-family: var(--font-mono);">${peekWordAnalysis.guess.toUpperCase()}</strong>
          <span style="font-size: 12px; color: var(--text-secondary); margin-left: 12px;">
            Entropy: ${peekWordAnalysis.entropy.toFixed(3)} bits • Exp. Left: ${peekWordAnalysis.expectedSize.toFixed(1)} • Worst: ${peekWordAnalysis.worstCaseSize}
          </span>
        </div>
        <button class="btn-table-action" data-play-word="${peekWordAnalysis.guess}">▶ Play</button>
      </div>
    ` : ''}

    <div class="table-container">
      ${ranked.length === 0 ? `
        <div class="empty-state" style="padding: 30px 16px;">
          <p style="color: var(--text-secondary); font-size: 12px;">No candidate guesses available for this state.</p>
        </div>
      ` : `
        <table class="strategy-table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Guess</th>
              <th>Entropy</th>
              <th>Exp. Left</th>
              <th>Worst Bin</th>
              <th>P(Ans)</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            ${ranked.slice(0, displayLimit).map((a, i) => {
              const isTreeOpt = decisionTreeOptimal && a.guess === decisionTreeOptimal;
              return `
                <tr style="${isTreeOpt ? 'background: var(--accent-green-subtle);' : ''}">
                  <td class="mono" style="color: var(--text-muted);">#${i + 1}</td>
                  <td class="mono" style="font-weight: 700;">
                    ${a.guess.toUpperCase()}
                    ${a.isPossibleSolution ? '<span class="tag-target">Target</span>' : ''}
                    ${isTreeOpt ? '<span class="metric-badge green" style="margin-left: 4px;">⭐ Tree Optimal</span>' : ''}
                  </td>
                  <td class="mono">${a.entropy.toFixed(3)}</td>
                  <td class="mono">${a.expectedSize.toFixed(1)}</td>
                  <td class="mono">${a.worstCaseSize}</td>
                  <td class="mono">${a.solutionProbability ? (a.solutionProbability * 100).toFixed(1) + '%' : '-'}</td>
                  <td>
                    <button class="btn-table-action" data-play-word="${a.guess}">▶ Play</button>
                  </td>
                </tr>
              `;
            }).join('')}
          </tbody>
        </table>
      `}
    </div>
  `;
}

function renderBinsTab(): string {
  const activeNode = engine.tree.activeNode;
  if (!activeNode) {
    return `
      <div style="padding: 30px; text-align: center; color: var(--text-muted); font-size: 13px;">
        No active move selected. Play a move or select one in the tree to inspect its score partition bins.
      </div>
    `;
  }

  const analysis = activeNode.userAnalysis;
  const buckets = analysis.buckets || {};
  const scores = Object.keys(buckets).map(Number).sort((a, b) => buckets[b].length - buckets[a].length);
  const total = activeNode.candidatesBefore.length;

  return `
    <div style="margin-bottom: 12px; font-size: 13px; color: var(--text-secondary);">
      Score partitions for <strong>${activeNode.guess.toUpperCase()}</strong> across <strong>${total}</strong> candidates:
    </div>

    <div class="buckets-list">
      ${scores.map(s => {
        const words = buckets[s] || [];
        const isActual = s === activeNode.score;
        const emoji = formatScoreEmoji(s);
        const pct = total > 0 ? ((words.length / total) * 100).toFixed(1) : '0';
        const isExpanded = expandedBucketScore === s;

        return `
          <div class="bucket-item ${isActual ? 'actual-outcome' : ''}" data-bucket-score="${s}">
            <div class="bucket-left">
              <span class="bucket-pattern">${emoji}</span>
              <div>
                <div>
                  <strong>${words.length}</strong> words (${pct}%)
                  ${isActual ? '<span class="metric-badge green" style="margin-left: 6px;">Actual Clue</span>' : ''}
                </div>
                <div class="bucket-words-sample">
                  ${isExpanded ? words.join(', ') : words.slice(0, 6).join(', ') + (words.length > 6 ? ` ... (+${words.length - 6} more)` : '')}
                </div>
              </div>
            </div>
            <div class="bucket-badge">${isExpanded ? '▲' : '▼'}</div>
          </div>
        `;
      }).join('')}
    </div>
  `;
}

function attachEventListeners() {
  // Mode switcher buttons
  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const mode = (btn as HTMLElement).dataset.mode as 'known' | 'unknown';
      engine.setMode(mode);
    });
  });

  // Tree view mode toggle
  document.querySelectorAll('[data-tree-view]').forEach(btn => {
    btn.addEventListener('click', () => {
      treeViewMode = (btn as HTMLElement).dataset.treeView as 'inline' | 'tree';
      render();
    });
  });

  // Target input
  const targetInput = document.getElementById('secret-target-input') as HTMLInputElement;
  if (targetInput) {
    targetInput.addEventListener('change', () => {
      const val = targetInput.value.trim().toLowerCase();
      if (val.length === 5 && wordList.allGuesses.includes(val)) {
        currentTypingWord = '';
        engine.setSecretSolution(val, true);
        showToast(`Target set to "${val.toUpperCase()}"`, 'info');
      } else if (val.length === 5) {
        showToast(`"${val.toUpperCase()}" is not in dictionary`, 'warning');
      }
    });
  }

  // Target bar buttons
  const btnToggleMask = document.getElementById('btn-toggle-mask');
  if (btnToggleMask) {
    btnToggleMask.addEventListener('click', () => engine.toggleHideSecret());
  }

  const btnRandomSecret = document.getElementById('btn-random-secret');
  if (btnRandomSecret) {
    btnRandomSecret.addEventListener('click', () => {
      currentTypingWord = '';
      engine.setRandomSecret(true);
      showToast('🎲 Selected new random secret word', 'info');
    });
  }

  const btnTodayWordle = document.getElementById('btn-today-wordle');
  if (btnTodayWordle) {
    btnTodayWordle.addEventListener('click', () => {
      const today = new Date();
      const dayIndex = Math.abs((today.getFullYear() * 365 + today.getMonth() * 31 + today.getDate()) % wordList.target.length);
      const todaysWord = wordList.target[dayIndex];
      currentTypingWord = '';
      engine.setSecretSolution(todaysWord, true);
      engine.hideSecret = true;
      showToast(`📅 Loaded Today's Wordle (${today.toISOString().split('T')[0]}) • Target is masked`, 'success');
      render();
    });
  }

  // Unknown mode state input
  const stateInputBox = document.getElementById('state-input-box') as HTMLInputElement;
  if (stateInputBox) {
    stateInputBox.addEventListener('input', () => {
      const res = validateStateString(stateInputBox.value, wordList);
      const pill = document.getElementById('state-feedback-pill');
      if (pill) {
        if (!res.isValid) {
          pill.className = 'state-feedback-tag error';
          pill.textContent = `❌ ${res.error || 'Invalid state'}`;
        } else if (res.isInconsistent) {
          pill.className = 'state-feedback-tag error';
          pill.textContent = '⚠️ 0 matching candidates';
        } else {
          pill.className = 'state-feedback-tag valid';
          pill.textContent = `🟢 ${res.candidateCount} candidates`;
        }
      }
    });
  }

  const btnLoadState = document.getElementById('btn-load-state');
  if (btnLoadState && stateInputBox) {
    btnLoadState.addEventListener('click', () => {
      try {
        engine.loadStateString(stateInputBox.value);
        showToast('🟢 Loaded custom state successfully', 'success');
      } catch (err) {
        showToast((err as Error).message, 'warning');
      }
    });
  }

  // Toolbar actions
  const btnPlayOpt = document.getElementById('btn-play-optimal');
  if (btnPlayOpt) {
    btnPlayOpt.addEventListener('click', async () => {
      if (isComputingOptimal || isAutoSolving || engine.activePath.length >= 6) return;
      if (engine.activeNode && engine.activeNode.score === 242) return;

      isComputingOptimal = true;
      currentTypingWord = '';
      render();

      await new Promise(resolve => setTimeout(resolve, 30));

      try {
        engine.playOptimalMove();
      } finally {
        isComputingOptimal = false;
        render();
      }
    });
  }

  const btnAutoSolve = document.getElementById('btn-auto-solve');
  if (btnAutoSolve) {
    btnAutoSolve.addEventListener('click', async () => {
      if (isAutoSolving) {
        autoSolveAbort = true;
        isAutoSolving = false;
        render();
        showToast('⏹ Auto-solve stopped', 'info');
        return;
      }

      if (engine.activePath.length >= 6) {
        showToast('Max 6 moves reached', 'warning');
        return;
      }
      if (engine.activeNode && engine.activeNode.score === 242) {
        showToast('Game is already solved!', 'success');
        return;
      }
      if (engine.mode === 'known' && !engine.secretSolution) {
        showToast('Please enter a target word first', 'warning');
        return;
      }

      isAutoSolving = true;
      autoSolveAbort = false;
      currentTypingWord = '';
      render();

      try {
        while (isAutoSolving && !autoSolveAbort) {
          if (engine.activePath.length >= 6) break;
          if (engine.activeNode && engine.activeNode.score === 242) break;
          if (engine.currentCandidates.length === 0) break;

          const nextNode = engine.playOptimalMove();
          if (!nextNode) break;

          render();

          if (nextNode.score === 242) {
            showToast(`🎉 Solved in ${engine.activePath.length} moves!`, 'success');
            break;
          }

          await new Promise(resolve => setTimeout(resolve, 320));
        }
      } finally {
        isAutoSolving = false;
        autoSolveAbort = false;
        render();
      }
    });
  }

  const btnToggleShowOpt = document.getElementById('btn-toggle-show-optimal');
  if (btnToggleShowOpt) {
    btnToggleShowOpt.addEventListener('click', () => engine.toggleShowOptimalCard());
  }

  const btnStepBack = document.getElementById('btn-step-back');
  if (btnStepBack) {
    btnStepBack.addEventListener('click', () => {
      currentTypingWord = '';
      engine.stepBack();
    });
  }

  const btnResetGame = document.getElementById('btn-reset-game');
  if (btnResetGame) {
    btnResetGame.addEventListener('click', () => {
      currentTypingWord = '';
      engine.reset();
      showToast('Board reset', 'info');
    });
  }

  // Theme toggle
  const btnTheme = document.getElementById('btn-theme-toggle');
  if (btnTheme) {
    btnTheme.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') || 'dark';
      document.documentElement.setAttribute('data-theme', current === 'dark' ? 'light' : 'dark');
    });
  }

  // Share URL button
  const btnShare = document.getElementById('btn-share-url');
  if (btnShare) {
    btnShare.addEventListener('click', () => {
      const url = new URL(window.location.href);
      if (engine.mode === 'known' && engine.secretSolution) {
        url.searchParams.set('target', engine.secretSolution);
      } else {
        url.searchParams.delete('target');
      }
      const st = engine.toStateString();
      if (st) url.searchParams.set('state', st);
      navigator.clipboard.writeText(url.toString());
      showToast('🔗 Game link copied to clipboard!', 'success');
    });
  }

  // Tile clicks in Unknown Word Mode
  document.querySelectorAll('.clickable-tile').forEach(tile => {
    tile.addEventListener('click', () => {
      const nodeId = (tile as HTMLElement).dataset.nodeId!;
      const letterIndex = parseInt((tile as HTMLElement).dataset.letterIndex!, 10);
      engine.updateTileColor(nodeId, letterIndex);
    });
  });

  // Move tree selections and promotions
  document.querySelectorAll('[data-select-node]').forEach(btn => {
    btn.addEventListener('click', () => {
      currentTypingWord = '';
      const nodeId = (btn as HTMLElement).dataset.selectNode!;
      engine.selectNode(nodeId);
    });
  });

  document.querySelectorAll('[data-promote-id]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const nodeId = (btn as HTMLElement).dataset.promoteId!;
      engine.promoteToMainLine(nodeId);
    });
  });

  document.querySelectorAll('[data-delete-id]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const nodeId = (btn as HTMLElement).dataset.deleteId!;
      engine.deleteNode(nodeId);
    });
  });

  // Tabs switching
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      activeRightTab = (btn as HTMLElement).dataset.tab as 'candidates' | 'topn' | 'bins';
      render();
    });
  });

  // Candidates search
  const candSearch = document.getElementById('candidate-search-input') as HTMLInputElement;
  if (candSearch) {
    candSearch.addEventListener('input', (e) => {
      candidateSearchQuery = (e.target as HTMLInputElement).value;
      const chipsContainer = document.querySelector('.candidate-chips-container');
      if (chipsContainer) {
        const pool = engine.currentCandidates;
        const filtered = candidateSearchQuery
          ? pool.filter(w => w.includes(candidateSearchQuery.toLowerCase()))
          : pool;
        chipsContainer.innerHTML = filtered.slice(0, 200).map(w => `
          <button class="candidate-chip" data-play-word="${w}">
            ${w.toUpperCase()}
          </button>
        `).join('');
        chipsContainer.querySelectorAll('[data-play-word]').forEach(chip => {
          chip.addEventListener('click', () => {
            currentTypingWord = '';
            const word = (chip as HTMLElement).dataset.playWord!;
            engine.addGuess(word);
          });
        });
      }
    });
  }

  // Play candidate chips
  document.querySelectorAll('[data-play-word]').forEach(chip => {
    chip.addEventListener('click', () => {
      currentTypingWord = '';
      const word = (chip as HTMLElement).dataset.playWord!;
      engine.addGuess(word);
    });
  });

  // Global Strategy & Weighted controls
  const stratSelect = document.getElementById('global-strategy-select') as HTMLSelectElement;
  if (stratSelect) {
    stratSelect.addEventListener('change', () => {
      engine.setStrategy(stratSelect.value);
    });
  }

  const weightedToggle = document.getElementById('global-weighted-toggle') as HTMLInputElement;
  if (weightedToggle) {
    weightedToggle.addEventListener('change', () => {
      engine.setWeighted(weightedToggle.checked);
    });
  }

  const topnLimitSelect = document.getElementById('topn-limit-select') as HTMLSelectElement;
  if (topnLimitSelect) {
    topnLimitSelect.addEventListener('change', () => {
      topNLimit = parseInt(topnLimitSelect.value, 10);
      render();
    });
  }

  // Peek word search
  const peekInput = document.getElementById('peek-word-input') as HTMLInputElement;
  const btnPeekSearch = document.getElementById('btn-peek-search');
  if (peekInput && btnPeekSearch) {
    const doPeek = () => {
      const w = peekInput.value.trim().toLowerCase();
      if (w.length === 5 && wordList.allGuesses.includes(w)) {
        peekWordQuery = w;
        peekWordAnalysis = analyze(w, engine.currentCandidates, engine.isWeighted ? wordList.weights : undefined, false);
      } else {
        peekWordAnalysis = null;
      }
      render();
    };
    btnPeekSearch.addEventListener('click', doPeek);
    peekInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') doPeek();
    });
  }

  // Bucket expand/collapse
  document.querySelectorAll('.bucket-item').forEach(item => {
    item.addEventListener('click', () => {
      const s = parseInt((item as HTMLElement).dataset.bucketScore!, 10);
      expandedBucketScore = expandedBucketScore === s ? null : s;
      render();
    });
  });

  // Virtual keyboard clicks
  document.querySelectorAll('.key-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = (btn as HTMLElement).dataset.key!;
      handleKeyPress(key);
    });
  });
}

init();
