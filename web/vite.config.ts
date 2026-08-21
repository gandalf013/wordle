import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    proxy: {
      '/api/nyt': {
        target: 'https://www.nytimes.com',
        changeOrigin: true,
        rewrite: (path) => {
          const url = new URL('http://localhost' + path);
          const date = url.searchParams.get('date') || new Date().toISOString().split('T')[0];
          return `/svc/wordle/v2/${date}.json`;
        },
      },
    },
  },
});
