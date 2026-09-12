# Deploying for free — Hugging Face Spaces (Docker)

This is the **best free deployment** for this stack. Every other free tier
(Render, Fly, Koyeb, Railway) caps memory at 256–512 MB, which cannot run
PyTorch + OpenCV + MediaPipe + librosa + TensorFlow together. HF Spaces **CPU
Basic** gives 2 vCPU / **16 GB RAM** / 50 GB disk with a public
`*.hf.space` URL and needs **no credit card**.

Caveats: CPU-only inference (fine — the models are reference
implementations), and free Spaces sleep after ~48 h of inactivity (waking
takes ~1 min).

## How it works

One Docker container runs everything, supervised by supervisord:

```
Daphne (ASGI :7860)  ── HTTP + WebSocket (same origin as the SPA)
Celery worker        ── processes uploads asynchronously
Redis                ── broker + Channels channel layer (in-container)
SQLite               ── embedded DB (no external Postgres needed)
```

The built React app is served same-origin by Django/WhiteNoise, so the
frontend's `/api` calls and `/ws` WebSocket streams work with zero CORS
configuration.

## Step-by-step checklist

### 1. Push the deploy files to your branch

```
Dockerfile, supervisord.conf, entrypoint.sh, .dockerignore,
README.md (Space frontmatter), DEPLOY.md
```

### 2. Create the Space

1. Go to <https://huggingface.co/new-space> (free account, no card).
2. **Space name:** e.g. `deepfake-detection`
3. **License:** MIT
4. **SDK:** pick **Docker** (it will read `sdk: docker` from this README).
5. **Hardware:** CPU basic (free). *Do not* pick a GPU — it is paid.
6. Create the Space. It comes with an empty git repo and a README.

### 3. Put the code in the Space

Either push your repo branch to the Space remote:

```bash
# after committing the deploy files on your branch:
git remote add hf https://huggingface.co/spaces/<your-username>/<space-name>
git push hf <your-branch>:main
```

or connect your GitHub repo in Space **Settings → Repository** and select the
branch. HF builds from the root `Dockerfile` in both cases.

### 4. Set the environment variables

Space **Settings → Variables and secrets** — add each as a secret:

| Variable | Value | Purpose |
|---|---|---|
| `DJANGO_SECRET_KEY` | `<random long string>` | required in production |
| `DJANGO_DEBUG` | `false` | production mode |
| `DJANGO_ALLOWED_HOSTS` | `*` | accept `.hf.space` hostnames |
| `DJANGO_DB_ENGINE` | `sqlite` | embedded DB, no external service |
| `DJANGO_SERVE_SPA` | `true` | serve the React app same-origin |
| `EXTERNAL_URL_ENABLED` | `true` | allow analyzing videos by URL |
| `HEATMAP_ENABLED` | `true` | Grad-CAM heatmap + overlay panels |
| `REPORT_ENABLED` | `true` | downloadable PDF reports |
| `CELERY_WORKER_CONCURRENCY` | `2` | worker capacity (2 vCPU) |
| `MAX_UPLOAD_SIZE_BYTES` | `30000000` | 30 MB cap — the Spaces proxy rejects bodies near 50 MB |

Generate a secret key: `python -c "import secrets; print(secrets.token_urlsafe(64))"`

### 5. Wait for the build

The first build downloads ~2–3 GB of wheels (PyTorch CPU, TensorFlow CPU,
OpenCV, librosa, …) and takes 10–20 minutes. The entrypoint runs
`migrate` automatically on first boot. The app is live at:

```
https://<your-username>-<space-name>.hf.space
```

### 6. Verify

- Open the URL — the upload page should render.
- Check Space **Settings → Logs** for the supervisord startup lines.
- Upload a short mp4 and watch the live pipeline + XAI panels.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Build fails on pip | Read the build log; most likely a version pin conflict — check `requirements.txt` against Python 3.11 wheels. |
| 400 Bad Request on load | `DJANGO_ALLOWED_HOSTS` missing or wrong; set it to `*`. |
| WebSocket won't connect | Same cause — the Channels origin validator uses `ALLOWED_HOSTS`. |
| Upload rejected 413 | The Space proxy caps request bodies (~50 MB); keep videos under the limit. |
| Slow first request after idle | Free Spaces sleep after ~48 h idle; waking takes ~1 min. |

## Alternatives

- **Oracle Cloud Always Free ARM VM** (4 vCPU / 24 GB, always on, real
  Postgres + Redis via docker-compose) — best if you want a permanent always-on
  server; needs an Oracle account with a card for verification (never charged).
- Render/Fly/Railway/Koyeb free tiers are **not viable**: 256–512 MB RAM
  cannot hold torch + TF + opencv + mediapipe simultaneously.