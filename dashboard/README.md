# stateset-agents Training Lab Dashboard

A React + Vite SPA for the `stateset-agents` RL training lab. Talks to the
FastAPI router at [`stateset_agents/api/routers/training_lab.py`](../stateset_agents/api/routers/training_lab.py)
under `/api/lab/*`, including a WebSocket stream for live training metrics.

## Features

| Surface | What it does |
|---|---|
| **Dashboard** | Lists experiments with search/filter/batch actions, sparklines, status pulses |
| **New Experiment** | 4-step wizard (Environment → Agent → Training → Review), with clone-from-existing |
| **Live Monitor** | WS-driven reward / loss / KL / entropy / LR charts with EMA smoothing, brush, episode browser, in-place config edits, log console, JSON export |
| **Compare** | Up to 6 experiments side-by-side: reward/loss series, radar of best/avg/convergence, config diff |
| **Playground** | Chat-style agent interaction with per-turn reward breakdown, saved sessions in localStorage |
| **Leaderboard** | Sortable rankings across runs |
| **Command Palette** | ⌘K / Ctrl+K — keyboard navigation across views |
| **Notifications** | Toast + bell center; status transitions and 75% milestones |

## Getting started

```bash
# from repo root, start the API
uvicorn stateset_agents.api.main:app --reload --port 8000

# in another shell
cd dashboard
npm install
npm run dev    # http://localhost:5173 — Vite proxies /api → :8000
```

Hotkeys: `⌘K` (palette), `N` (new), `D` (dashboard), `P` (playground),
`L` (leaderboard), `C` (compare).

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Vite dev server on port 5173 with HMR; proxies `/api` and `/api/.../ws` to `http://localhost:8000` |
| `npm run build` | `tsc -b && vite build` — type-checks then builds to `dist/` |
| `npm run lint` | ESLint over `src/` |
| `npm run typecheck` | `tsc --noEmit` |
| `npm test` | Vitest (jsdom + Testing Library) |

## Architecture

```
src/
├── App.tsx                       view switcher + experiment polling
├── api.ts                        typed fetch wrapper for /api/lab/*
├── types.ts                      shared TS types matching the API schema
├── components/
│   ├── Dashboard.tsx             experiment list + search/filter/batch
│   ├── LiveMonitor.tsx           tabbed monitor (overview/episodes/config)
│   ├── CreateExperiment.tsx      4-step wizard
│   ├── CompareExperiments.tsx    multi-run comparison
│   ├── Playground.tsx            chat-style agent interaction
│   ├── Leaderboard.tsx           sortable rankings
│   ├── MetricsCharts.tsx         recharts wrappers w/ EMA smoothing
│   ├── EpisodeBrowser.tsx        paginated table + filters
│   ├── ExperimentDrawer.tsx      quick-view drawer
│   ├── CommandPalette.tsx        ⌘K dialog (focus-trapped, ARIA-conformant)
│   ├── NotificationCenter.tsx    bell + dropdown, backed by external store
│   ├── ToastProvider.tsx         portal-style toast region
│   ├── TrainingConsole.tsx       live log tail with level/search filters
│   ├── Layout.tsx                sidebar shell
│   └── ...                       Card, Skeleton, Onboarding, etc.
├── hooks/
│   ├── useToast.ts                       toast API (success/error/info)
│   ├── toast-context.ts                  toast React context
│   ├── useExperimentWs.ts                metrics + latest-episode WebSocket
│   ├── useHotkeys.ts                     single-key shortcuts
│   ├── notifications-store.ts            useSyncExternalStore-backed log
│   └── useTrackExperimentNotifications.ts effect that diffs status transitions
└── __tests__/                    Vitest smoke tests
```

### Data flow

- **Polling** (5 s) for the experiment list in `App.tsx`. Live monitor reverts
  to polling only when the WS is closed (3 s).
- **WebSocket** at `/api/lab/experiments/{id}/ws` — pushes `metrics` and
  `episode` events while a run is active. 30 s keepalive ping from the client.
- **Notifications** are pushed to a module-level store via
  `useSyncExternalStore`, so the bell survives component unmounts and the
  tracking hook can derive notifications without `setState`-in-effect.

### Styling

CSS variables in `src/index.css` (zinc neutrals + indigo accent). Components
use inline styles by design — the surface is small enough that a CSS-in-JS or
Tailwind dependency would be overkill. Animations live in `src/App.css`.

## Code-quality bar

CI runs lint + typecheck + build + tests on every PR touching `dashboard/`.
See `.github/workflows/dashboard.yml`.

- `tsc --strict`, `verbatimModuleSyntax: true`
- `react-hooks/exhaustive-deps` and `react-hooks/set-state-in-effect` enforced
- Toast region is `aria-live="polite"`; palette is a focus-trapped dialog
- Notifications use an external store (`useSyncExternalStore`) so component
  remounts don't drop history

## Production build

```bash
npm run build
# Serve dist/ behind any static host; reverse-proxy /api → FastAPI
```

Manual chunks split `recharts` and `lucide-react` for cache friendliness
(see `vite.config.ts`).
