/**
 * Central Branding & Site Configuration
 * 
 * Edit this file directly to update the application name, tagline, logo tiles,
 * disclaimer, domain, and metadata without needing code changes.
 */

export interface LogoTileConfig {
  letter: string;
  color: 'green' | 'yellow' | 'gray';
}

export interface AppConfig {
  name: string;
  shortName: string;
  tagline: string;
  domain: string;
  logoTiles: LogoTileConfig[];
  disclaimer: string;
  meta: {
    title: string;
    description: string;
  };
  links: {
    github?: string;
    docs?: string;
  };
}

export const APP_CONFIG: AppConfig = {
  // Primary application name
  name: 'Word Explorer',
  
  // Short acronym/identifier
  shortName: 'WEX',
  
  // Tagline displayed in header and share cards
  tagline: 'Interactive Explorer & Strategy Engine for Wordle',
  
  // Production domain or deployment URL
  domain: 'https://wex.pages.dev',
  
  // 3-letter header logo badge tiles and their background colors
  logoTiles: [
    { letter: 'W', color: 'green' },
    { letter: 'E', color: 'yellow' },
    { letter: 'X', color: 'gray' },
  ],
  
  // Legal nominative fair-use disclaimer displayed in the footer
  disclaimer:
    'WEX is an independent strategy and analysis tool. It is not affiliated with, sponsored by, or endorsed by The New York Times Company. Wordle is a registered trademark of The New York Times Company.',
  
  meta: {
    title: 'Word Explorer (WEX) — Explorer for Wordle',
    description:
      'High-performance solver, WordleBot skill/luck analysis, what-if branching, and decision tree explorer for Wordle.',
  },
  
  links: {
    github: 'https://github.com',
  },
};
