# ArchibaldChain.github.io

Personal website built with Astro.

## Development

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
```

The build writes the static site to `dist/`.

## Editing Content

- English home page: `src/pages/index.astro`
- Chinese home page: `src/pages/Chinese.astro`
- Project detail pages: add Markdown files in `src/content/projects/`
- Articles/blog posts: add Markdown files in `src/content/articles/`
- Shared layout and reusable UI: `src/layouts/` and `src/components/`
- Site behavior source: `src/stisla.ts`

The browser script is compiled into `public/js/stisla.js` during `npm run build`.
