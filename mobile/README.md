# stateset-agents Mobile Training Lab

An Expo Router app for browsing and launching `stateset-agents` training
runs from a phone. Talks to the same FastAPI `/api/lab/*` router as the
[`dashboard/`](../dashboard) app (`stateset_agents/api/routers/training_lab.py`).

## Status: demo, not deployed

This app is real, working code, but it has **no deployment path today** —
it is not published to the App Store / Play Store or Expo's hosting, and
there is no CI/CD pipeline that ships it anywhere.

- The `/api/lab/*` backend it talks to is **simulator-backed**: it
  scripts a simulation of training rather than running real GRPO/GSPO
  jobs.
- That backend is gated behind auth *and* the `API_ENABLE_TRAINING_LAB`
  server flag, off by default outside local development
  (`stateset_agents/api/config.py`).
- **`mobile/hooks/useTrainingData.ts` silently falls back to bundled mock
  data** (`mobile/lib/mockData.ts`) whenever the API is unreachable,
  unauthenticated, or returns an empty result — which, given the above,
  is the default state for anyone who hasn't stood up a local API. The
  hook now exposes `source: 'live' | 'mock'` and a convenience
  `isMockData: boolean` on its return value, and logs one
  `console.warn` the first time it falls back per app session, so the
  fallback is visible instead of indistinguishable from real data.
- Treat this as a local dev tool / internal demo, not a production app,
  until someone explicitly builds and ships it.

## Getting started

```bash
# from repo root, start the API
uvicorn stateset_agents.api.main:app --reload --port 8000
# ensure API_ENABLE_TRAINING_LAB=true and valid credentials are configured,
# otherwise the app will run entirely on mock data

# in another shell
cd mobile
npm install
npm start          # Expo dev server; press i/a/w for iOS/Android/web
```

Point the app at a non-default API host with `EXPO_PUBLIC_API_BASE_URL`
(see `mobile/lib/api.ts`; defaults to `http://10.0.2.2:8000`, the Android
emulator's alias for the host machine's `localhost`). Set
`EXPO_PUBLIC_API_KEY` to send it as the `X-API-Key` header on every request.

## Scripts

| Command | What it does |
|---|---|
| `npm start` | Expo dev server |
| `npm run ios` / `npm run android` / `npm run web` | Launch on a specific platform |
| `npm run typecheck` | `tsc --noEmit` |

There is no `npm test` script defined yet.

## Architecture

```
app/            expo-router screens (dashboard, runs, models, datasets, more)
components/     training/, ui/ presentational components
hooks/
├── useTrainingData.ts   React Query wrapper: live API with mock fallback
lib/
├── api.ts               typed fetch wrapper for /api/lab/*
├── mockData.ts           bundled demo runs/datasets/models/algorithms
├── types.ts              shared TS types
└── format.ts             small formatting helpers
theme/          design tokens
```

## What productionizing would need

- A hosted FastAPI deployment with `API_ENABLE_TRAINING_LAB=true`, real
  auth, and CORS/rate-limit review for mobile clients.
- Replacing (or explicitly keeping, with a clear in-app label) the
  simulator backend with real training-job orchestration.
- A build pipeline (EAS Build or equivalent) and App Store / Play Store
  (or internal distribution) release process — none exists today.
- Deciding whether the mock-data fallback should remain a silent
  degrade-gracefully feature (with the new `isMockData` flag surfaced in
  the UI as a badge) or be removed once a backend is always expected to
  be reachable.
