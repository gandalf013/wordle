export const Score = {
  GRAY: 0,
  YELLOW: 1,
  GREEN: 2,
} as const;

export type Score = typeof Score[keyof typeof Score];

export const EMOJI_MAP: Record<Score, string> = {
  [Score.GRAY]: '⬛',
  [Score.YELLOW]: '🟨',
  [Score.GREEN]: '🟩',
};

export function getScore(guess: string, target: string): number {
  if (guess.length === 5 && target.length === 5) {
    const g0 = guess.charCodeAt(0), g1 = guess.charCodeAt(1), g2 = guess.charCodeAt(2), g3 = guess.charCodeAt(3), g4 = guess.charCodeAt(4);
    const t0 = target.charCodeAt(0), t1 = target.charCodeAt(1), t2 = target.charCodeAt(2), t3 = target.charCodeAt(3), t4 = target.charCodeAt(4);

    let s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
    let u0 = 1, u1 = 1, u2 = 1, u3 = 1, u4 = 1;

    // First pass: Greens
    if (g0 === t0) { s0 = 2; u0 = 0; }
    if (g1 === t1) { s1 = 2; u1 = 0; }
    if (g2 === t2) { s2 = 2; u2 = 0; }
    if (g3 === t3) { s3 = 2; u3 = 0; }
    if (g4 === t4) { s4 = 2; u4 = 0; }

    // Second pass: Yellows
    if (s0 === 0) {
      if (u0 && g0 === t0) { s0 = 1; u0 = 0; }
      else if (u1 && g0 === t1) { s0 = 1; u1 = 0; }
      else if (u2 && g0 === t2) { s0 = 1; u2 = 0; }
      else if (u3 && g0 === t3) { s0 = 1; u3 = 0; }
      else if (u4 && g0 === t4) { s0 = 1; u4 = 0; }
    }
    if (s1 === 0) {
      if (u0 && g1 === t0) { s1 = 1; u0 = 0; }
      else if (u1 && g1 === t1) { s1 = 1; u1 = 0; }
      else if (u2 && g1 === t2) { s1 = 1; u2 = 0; }
      else if (u3 && g1 === t3) { s1 = 1; u3 = 0; }
      else if (u4 && g1 === t4) { s1 = 1; u4 = 0; }
    }
    if (s2 === 0) {
      if (u0 && g2 === t0) { s2 = 1; u0 = 0; }
      else if (u1 && g2 === t1) { s2 = 1; u1 = 0; }
      else if (u2 && g2 === t2) { s2 = 1; u2 = 0; }
      else if (u3 && g2 === t3) { s2 = 1; u3 = 0; }
      else if (u4 && g2 === t4) { s2 = 1; u4 = 0; }
    }
    if (s3 === 0) {
      if (u0 && g3 === t0) { s3 = 1; u0 = 0; }
      else if (u1 && g3 === t1) { s3 = 1; u1 = 0; }
      else if (u2 && g3 === t2) { s3 = 1; u2 = 0; }
      else if (u3 && g3 === t3) { s3 = 1; u3 = 0; }
      else if (u4 && g3 === t4) { s3 = 1; u4 = 0; }
    }
    if (s4 === 0) {
      if (u0 && g4 === t0) { s4 = 1; u0 = 0; }
      else if (u1 && g4 === t1) { s4 = 1; u1 = 0; }
      else if (u2 && g4 === t2) { s4 = 1; u2 = 0; }
      else if (u3 && g4 === t3) { s4 = 1; u3 = 0; }
      else if (u4 && g4 === t4) { s4 = 1; u4 = 0; }
    }

    return s0 * 81 + s1 * 27 + s2 * 9 + s3 * 3 + s4;
  }

  if (guess.length !== target.length) {
    throw new Error(`Guess ${guess} not valid for target ${target}`);
  }

  const n = guess.length;
  const counts: Record<string, number> = {};
  for (let i = 0; i < n; i++) {
    const ch = target[i];
    counts[ch] = (counts[ch] || 0) + 1;
  }

  const score: Score[] = new Array(n).fill(Score.GRAY);

  // First pass: Greens
  for (let i = 0; i < n; i++) {
    if (guess[i] === target[i]) {
      score[i] = Score.GREEN;
      counts[guess[i]]--;
    }
  }

  // Second pass: Yellows
  for (let i = 0; i < n; i++) {
    if (score[i] !== Score.GREEN && (counts[guess[i]] || 0) > 0) {
      score[i] = Score.YELLOW;
      counts[guess[i]]--;
    }
  }

  return getScoreNum(score);
}

export function getScoreNum(score: Score[]): number {
  const n = score.length;
  let total = 0;
  for (let i = 0; i < n; i++) {
    total += Math.pow(3, n - i - 1) * score[i];
  }
  return total;
}

export function getScoreList(scoreNum: number, n: number = 5): Score[] {
  if (scoreNum < 0 || scoreNum >= Math.pow(3, n)) {
    throw new Error(`Score ${scoreNum} out of bounds for word length ${n}`);
  }
  const r: Score[] = [];
  let s = scoreNum;
  while (s > 0) {
    const rem = s % 3;
    r.push(rem as Score);
    s = Math.floor(s / 3);
  }
  while (r.length < n) {
    r.push(Score.GRAY);
  }
  return r.reverse();
}

export function formatScoreEmoji(scoreNum: number, n: number = 5): string {
  const list = getScoreList(scoreNum, n);
  return list.map(s => EMOJI_MAP[s]).join('');
}

export function parseScoreString(str: string, n: number = 5): number {
  const cleaned = str.trim();
  if (cleaned.length !== n) {
    throw new Error(`Invalid score string length: ${cleaned}`);
  }

  const scores: Score[] = [];
  for (let i = 0; i < n; i++) {
    const ch = cleaned[i];
    if (ch === '0' || ch === 'b' || ch === 'B' || ch === '⬛' || ch === '⬜') {
      scores.push(Score.GRAY);
    } else if (ch === '1' || ch === 'y' || ch === 'Y' || ch === '🟨') {
      scores.push(Score.YELLOW);
    } else if (ch === '2' || ch === 'g' || ch === 'G' || ch === '🟩') {
      scores.push(Score.GREEN);
    } else {
      throw new Error(`Unknown score character: ${ch}`);
    }
  }
  return getScoreNum(scores);
}
