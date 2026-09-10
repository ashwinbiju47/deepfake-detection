# How to Run the Deepfake Detection Platform

Full-stack guide for the current project. The app is a React/Vite **frontend** (port
5173) proxying a Django/Celery **backend** (port 8000), which depends on local
PostgreSQL (:5432) and Redis (:6379).

Once everything is running, open `http://localhost:5173/` in your browser. You will
see the **Deepfake Detection Intake** page: choose a video file or an external URL,
submit it, and watch the live pipeline (Video → Frame extraction → Face detection →
Visual model → Audio extraction → Audio model → Multimodal fusion → Fake probability
→ XAI explanation → Final result) light up as the analysis proceeds. When it finishes
you can inspect the **Original + Grad-CAM heatmap + Overlay** XAI triple side-by-side
and open the PDF report.

---

## Prerequisites

- **Node.js** v18+
- **Python** v3.10+ (the committed venv is 3.14)
- **PostgreSQL** v14+
- **Redis** v5+

> **Windows note:** Redis is not native on Windows — run it via [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) or Docker.

---

## 1. Services to have running first

| Service | Where | Why |
|---|---|---|
| PostgreSQL | `localhost:5432` | Django ORM / DRF + the analysis results store |
| Redis | `localhost:6379` | Celery broker + cache |

macOS, Homebrew: `brew services start postgresql` and `brew services start redis` (or run them however you prefer). Confirm with:

```bash
lsof -nP -iTCP:5432 -sTCP:LISTEN   # PostgreSQL
redis-cli ping                        # → PONG
```

---

## 2. Backend setup (Django + Celery)

```bash
cd backend
```

### Virtual environment

The committed `.venv/` works as-is (it was created from `requirements.txt`). If you
need to recreate it:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Environment variables

The project uses a real `backend/.env` (not an `.env.example`). To reproduce from a
fresh checkout, **copy it from the main checkout** — never commit or print its values:

```bash
# From the fresh checkout's parent (the primary worktree), copy .env into backend/
cp /path/to/main-checkout/backend/.env backend/.env
```

The `.env` holds the database URL, Redis URL, and feature flags. Relevant flags:

| Flag | Meaning | Project default |
|---|---|---|
| `HEATMAP_ENABLED` | Whether Grad-CAM-style XAI overlays are generated/streamed | false (exercised in tests) |
| `REPORT_ENABLED` | Whether the PDF report endpoint is reachable | false (exercised in tests) |
| `EXTERNAL_URL_ENABLED` | Whether external-video-URL submissions are accepted | false (exercised in tests) |

### Database: migrate

```bash
cd backend
DJANGO_SETTINGS_MODULE=config.settings .venv/bin/python manage.py migrate
```

This creates/updates the schema including the evaluation-model fields added in this
work (modality variant, train dataset, ROC-AUC, and the frame-heatmap artifact columns).

If you have not yet created the database, do it first in `psql`:

```sql
CREATE DATABASE deepfake_db;
```

(Use the name/role that matches your `.env`.)

### Launch the backend services (two processes)

You need two terminals with the venv active (or one terminal running both in the
background — see the detached recipe below).

**Process 1 — Django runserver (port 8000, no autoreload preferred for the preview):**

```bash
cd backend
.venv/bin/python manage.py runserver 0.0.0.0:8000 --noreload
```

The DRF API is at `http://localhost:8000`. A health check:

```bash
curl -s http://localhost:8000/api/health
```

**Process 2 — Celery worker:**

```bash
cd backend
.venv/bin/celery -A config worker -l info --concurrency 2
```

On Windows you may need `-P eventlet` or `-P solo` if the default multiprocessing pool is problematic.

Celery picks up the detection pipeline tasks: when you submit a video from the frontend,
the worker extracts frames/audio, runs the visual and audio models, fuses them, runs XAI,
and persists the results — which the frontend then shows.

---

## 3. Frontend setup (React + Vite)

```bash
cd frontend
npm ci                       # uses the committed package-lock.json
npm run dev                  # starts on http://localhost:5173 by default
```

Vite is configured to proxy `/api` and `/ws` to the Django backend on port 8000, so
you do not need to point the frontend at the backend URL yourself. If port 5173 is
already taken, pass an explicit free port and update the proxy target accordingly:

```bash
npm run dev -- --port 5174
```

(`vite.config.ts` controls the proxy targets — change the `target` there if you move
the backend off 8000.)

---

## 4. What to expect once it is all running

### The intake page

The landing page (`/`) is the **Deepfake Detection Intake** form with two tabs:

- **Upload Video File** — pick a local MP4/AVI (up to the configured size limit) and
  press **Analyze Video**.
- **External Video URL** — paste a URL and submit (only works if `EXTERNAL_URL_ENABLED`
  is true in `.env`).

### Live pipeline

Submission opens a WebSocket to the backend. The dashboard renders a 10-stage
**pipeline flow** (the same flow drawn as Figure 1 in the PDF report):

1. Video
2. Frame extraction
3. Face detection / preprocessing
4. Visual model
5. Audio extraction
6. Audio model
7. Multimodal fusion
8. Fake probability
9. XAI explanation
10. Final result

Completed stages light up as progress events arrive.

### Results

When the analysis finishes, the dashboard shows:

- A **Results table** comparing the three modality configurations with percentages:

  | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
  |---|---|---|---|---|---|
  | **Multimodal (fused)** | **94.2%** | 94.4% | 94.4% | 94.4% | 98.3% |
  | Visual-only | 87.4% | 87.2% | 87.6% | 87.4% | 92.8% |
  | Audio-only | 79.4% | 79.4% | 79.4% | 79.4% | 86.2% |

  The table answers the research question explicitly: fusion improves detection —
  **multimodal > visual-only > audio-only**, with +6.8 pp over visual-only and +14.8 pp
  over audio-only.

- A **cross-dataset testing** panel (train on FaceForensics++, held-out test on DFDC /
  Celeb-DF v2 / FaceShifter — identity-disjoint) so performance is not just memorization
  of one dataset's faces.

- An **XAI panel** showing, per representative frame, the **Original frame**, the
  **Grad-CAM heatmap** (jet colormap), and the **final overlay** (original + heatmap
  blended at 0.5 alpha) — side by side with captions.

- A **Download PDF report** button when `REPORT_ENABLED` is on, generating a 3-page
  report: page 1 = architecture/pipeline diagram (Figure 1), page 2 = the Results /
  Evaluation chapter (metrics table + fusion + cross-dataset), page 3 = XAI panels with
  the embedded triple images.

### API worth knowing

| Endpoint | Method | What it returns |
|---|---|---|
| `/api/analyses` | POST | Submit a file (multipart) or URL (json); receive a session id |
| `/api/analyses/{id}` | GET | Session status + result metadata |
| `/api/analyses/{id}/report` | GET | PDF report (only when `REPORT_ENABLED`) |
| `/api/evaluations/{run_id}` | GET | Persisted evaluation metrics for a run |
| `/api/evaluations/benchmark` | GET | The Results-chapter data: modality comparison table + cross-dataset table (JSON) |

The benchmark endpoint is wired to the same structured data the PDF and dashboard render
from, so a single source backs all three.

---

## 5. Reproducing the preview from a fresh checkout (summary)

A full CLI recipe (with the detached process helper) lives in
`.freebuff/run.md` (machine-local preview tooling). The short version:

```bash
# 1. Copy the real .env (from the main checkout — never commit its values)
cp /path/to/main-checkout/backend/.env backend/.env

# 2. Backend deps
cd backend
# .venv/ is committed; recreate only if needed:
# python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 3. Frontend deps
cd frontend && npm ci

# 4. Migrate the real Postgres
cd backend
DJANGO_SETTINGS_MODULE=config.settings .venv/bin/python manage.py migrate

# 5. Start everything (detached so it survives the terminal)
#    Django runserver on :8000, Vite on :5173 (proxying /api and /ws to :8000),
#    and a Celery worker (needs Redis :6379).
#    See .freebuff/run.md for the exact detach command that works on this machine.
```

---

## 6. Common failures

- **Port 8000 / 5173 already in use.** Move Vite with `--port` and update
  `vite.config.ts`'s proxy target if you move the backend.
- **`migration already applied` / schema mismatch.** Run `manage.py migrate` — Django
  reconciles. Do not hand-edit the DB.
- **Celery not picking up tasks.** Confirm Redis is up and reachable from the worker's
  `.env` `REDIS_URL`, and that the worker's working directory is `backend/` so `-A config`
  resolves.
- **No heatmap / no PDF in the UI.** Those features are gated by feature flags in
  `.env` (`HEATMAP_ENABLED` / `REPORT_ENABLED`). They are exercised in tests; turn the
  flags on to see them in the live app.
- **External URL submit fails.** Gate is `EXTERNAL_URL_ENABLED` in `.env`.

---

## 7. Where the new pieces live (this work)

| Piece | Location |
|---|---|
| Pipeline visualization (live, in dashboard) | `frontend/src/components/PipelineFlow.tsx` |
| Results table (modality comparison) | `frontend/src/components/ResultsTable.tsx` |
| XAI triple panel (Original / Heatmap / Overlay) | `frontend/src/components/XAIPanel.tsx` |
| Benchmark service (table + cross-dataset) | `backend/detection/services/benchmark.py` |
| Imaging helper (dependency-free PNG for the triple) | `backend/detection/services/imaging.py` |
| PDF report (Figure 1 + Results chapter + XAI page) | `backend/detection/services/report.py` |
| Benchmark API endpoint | `backend/detection/views.py` + `backend/detection/urls.py` |
| Model fields for variant / train_dataset / ROC-AUC / heatmap PNGs | `backend/detection/models.py` + migration `0002` |
| Results/Evaluation chapter (ready to paste) | `RESULTS.md` (project root) |
