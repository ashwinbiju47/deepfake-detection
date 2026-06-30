# Deepfake Detection Platform — Frontend

React + TypeScript dashboard scaffold for the Real-Time Deepfake Detection
Platform. Built with Vite, styled with TailwindCSS, and visualized with
Chart.js / D3. Property-based tests use Vitest + fast-check.

## Scripts

- `npm run dev` — start the Vite dev server (proxies `/api` and `/ws` to the
  Django backend at `localhost:8000`)
- `npm run build` — type-check and build for production
- `npm run typecheck` — type-check only
- `npm run test` — run the test suite once (Vitest)
- `npm run test:watch` — run tests in watch mode

## Structure

- `src/api/` — typed REST client stub for the DRF endpoints
  (`POST /api/analyses`, `GET /api/analyses/{id}`, `.../report`,
  `GET /api/evaluations/{run_id}`) and shared backend contract types.
- `src/ws/` — typed WebSocket client stub for the Result_Streamer
  (`/ws/analyses/{session_id}/`), including the message protocol types and
  reconnect/resume scaffolding.
- `src/test/` — Vitest setup (enforces ≥100 fast-check iterations) and a
  tooling smoke test.

The full dashboard UI (upload form, progress charts, heatmap overlays, report
download) is implemented in a later task.
