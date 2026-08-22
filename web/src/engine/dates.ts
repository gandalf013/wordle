/**
 * Date utilities for Wordle date handling.
 * Wordle games are tied to the player's local calendar day.
 */

/**
 * Returns a date formatted as YYYY-MM-DD in the user's local timezone.
 *
 * @param d Optional Date object (defaults to current time)
 * @returns YYYY-MM-DD string in local time
 */
export function getLocalDateString(d: Date = new Date()): string {
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

/**
 * Steps a YYYY-MM-DD date string forward or backward by deltaDays.
 *
 * @param currentDateStr Date string formatted as YYYY-MM-DD
 * @param deltaDays Number of days to add (positive) or subtract (negative)
 * @returns YYYY-MM-DD string
 */
export function stepDate(currentDateStr: string, deltaDays: number): string {
  const d = new Date(currentDateStr + 'T00:00:00Z');
  d.setUTCDate(d.getUTCDate() + deltaDays);
  return d.toISOString().split('T')[0];
}
