import { describe, it, expect } from 'vitest';
import { getLocalDateString, stepDate } from './dates';

describe('Date Utilities for Wordle', () => {
  it('formats local Date object as YYYY-MM-DD using local time components', () => {
    // Construct dates in local time
    const lateNightDate = new Date(2026, 7, 21, 23, 45, 0); // Aug 21, 2026 23:45 local
    expect(getLocalDateString(lateNightDate)).toBe('2026-08-21');

    const earlyMorningDate = new Date(2026, 7, 22, 0, 15, 0); // Aug 22, 2026 00:15 local
    expect(getLocalDateString(earlyMorningDate)).toBe('2026-08-22');

    const singleDigitMonthAndDay = new Date(2026, 0, 5, 12, 0, 0); // Jan 5, 2026 local
    expect(getLocalDateString(singleDigitMonthAndDay)).toBe('2026-01-05');
  });

  it('steps date forward and backward across day/month/year boundaries', () => {
    expect(stepDate('2026-08-21', 1)).toBe('2026-08-22');
    expect(stepDate('2026-08-21', -1)).toBe('2026-08-20');

    // Month boundary
    expect(stepDate('2026-08-01', -1)).toBe('2026-07-31');
    expect(stepDate('2026-07-31', 1)).toBe('2026-08-01');

    // Year boundary
    expect(stepDate('2026-01-01', -1)).toBe('2025-12-31');
    expect(stepDate('2025-12-31', 1)).toBe('2026-01-01');

    // Leap year (2024 is a leap year)
    expect(stepDate('2024-02-28', 1)).toBe('2024-02-29');
    expect(stepDate('2024-02-29', 1)).toBe('2024-03-01');
    expect(stepDate('2024-03-01', -1)).toBe('2024-02-29');

    // Non-leap year (2025 is not a leap year)
    expect(stepDate('2025-02-28', 1)).toBe('2025-03-01');
    expect(stepDate('2025-03-01', -1)).toBe('2025-02-28');
  });
});
