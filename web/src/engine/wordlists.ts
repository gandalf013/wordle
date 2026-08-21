export interface WordList {
  target: string[];
  extra: string[];
  allGuesses: string[];
  wordLength: number;
  weights: Record<string, number>;
}

export function parseWordListText(text: string): WordList {
  const lines = text.split(/\r?\n/);
  const target: string[] = [];
  const extra: string[] = [];
  const weights: Record<string, number> = {};
  let r = target;
  let wordLength: number | null = null;

  for (const rawLine of lines) {
    const data = rawLine.trim();
    if (!data) {
      if (r === extra) {
        // already switched, ignore subsequent empty lines
        continue;
      }
      r = extra;
      continue;
    }

    const parts = data.split(/\s+/);
    const word = parts[0].toLowerCase();
    if (wordLength === null) {
      wordLength = word.length;
    } else if (word.length !== wordLength) {
      continue;
    }

    r.push(word);
    weights[word] = parts.length > 1 ? parseFloat(parts[1]) : 1.0;
  }

  const allGuesses = Array.from(new Set([...target, ...extra])).sort();

  return {
    target,
    extra,
    allGuesses,
    wordLength: wordLength || 5,
    weights,
  };
}

let cachedWordList: WordList | null = null;

export async function loadWordList(): Promise<WordList> {
  if (cachedWordList) return cachedWordList;
  const res = await fetch('/data/words.txt');
  if (!res.ok) {
    throw new Error(`Failed to load /data/words.txt: ${res.statusText}`);
  }
  const text = await res.text();
  cachedWordList = parseWordListText(text);
  return cachedWordList;
}
