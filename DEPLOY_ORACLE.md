# Deploying for free — Oracle Cloud Always Free ARM VM

**The genuinely free, always-on option for this stack.** Oracle's Always Free
tier gives a 4-core ARM VM with **24 GB RAM / 200 GB disk** — the only free
host anywhere that can run PyTorch + OpenCV + MediaPipe + TensorFlow together
(HF Spaces dropped free Docker hosting in July 2026; Render/Fly/Railway cap at
256–512 MB and OOM).

Cost: **$0 forever** (the 4 OCPU / 24 GB A1.Flex shape is within Always Free
limits). Oracle asks for a credit card at signup **for identity verification
only — it is never charged** unless you explicitly upgrade to a paid shape.

Time: ~30–45 min one-time (account + VM creation), then one command deploys.

---

## Part A — You create the account and the VM (~30 min)

These steps need your identity and card, so only you can do them. Follow
exactly:

### 1. Sign up

1. Go to <https://signup.oraclecloud.com> and create an account
   (email verification → password → home region).
2. Add the credit card when asked — **verification only, not charged**.
3. You may need to wait ~5–15 min for the account to be provisioned.

### 2. Create the Always Free ARM VM

1. Open **Compute → Instances → Create instance**.
2. **Name:** `deepfake`
3. **Image:** Ubuntu 22.04 (or 24.04) — **ARM** (the default A1 shape is ARM;
   do not switch to x86 or the 4 OCPU / 24 GB config exceeds the free limit).
4. **Shape:** *VM.Standard.A1.Flex* → **4 OCPU / 24 GB** (stays green = free).
5. **SSH keys:** paste the **public key** from the file
   `.freebuff/oracle_ssh_key.pub` in this project (see Part B step 1 below —
   generate it first, or paste your own key if you prefer).
6. Create the instance, then note its **Public IP** from the instance page.

### 3. Open the firewall (ingress rules)

1. Open **Networking → Virtual cloud networks → your VCN → Security Lists →
   Default Security List → Add Ingress Rules**.
2. Add a rule: **Source CIDR** `0.0.0.0/0`, **Destination Port** `80` (TCP).
3. (SSH on 22 is already open by default.)

> Port 80 serves the app directly. Skip HTTPS for now — free certs are a
> follow-up (`caddy` or a load balancer), not required to go live.

---

## Part B — I deploy it for you (or one command)

### Option 1 — I do it (recommended)

1. I generate an SSH keypair for you (or reuse one you provide) — public key
   pasted in Part A step 2.5 above.
2. You tell me the **public IP** and where the private key is
   (e.g. `.freebuff/oracle_ssh_key`).
3. I run the bootstrap remotely: install Docker, clone the repo, generate the
   secret key, `docker compose up -d --build`, and verify
   `http://<ip>/api/health` + the page.

### Option 2 — One command yourself

```bash
# on the VM (as the ubuntu user):
curl -fsSL https://raw.githubusercontent.com/ashwinbiju47/deepfake-detection/freebuff/need-to-change-4dac48fb-bf8f-4f32-800f-d43831556497/deploy/oracle_bootstrap.sh -o bootstrap.sh
bash bootstrap.sh
```

The script installs Docker, clones this branch, writes `backend/.env` with a
freshly generated `DJANGO_SECRET_KEY`, and starts the stack. First build
downloads ~2–3 GB of wheels — allow 15–25 minutes. It prints
`http://<public-ip>/` when done.

---

## Architecture on the VM

```
        Port 80 ──▶ web container (Dockerfile, port 7860)
                      ├── Daphne   HTTP + WebSocket (same origin as the SPA)
                      ├── Celery   async analysis worker
                      └── Redis    broker + Channels channel layer (in-container)
        Port 5432 ◀── db container — PostgreSQL 16 (named volume pgdata)
```

- The built React app is served same-origin by Django/WhiteNoise — zero CORS.
- `entrypoint.sh` runs `migrate` automatically on every boot.
- Data survives rebuilds (`pgdata` volume); transient uploads are purged after
  analysis by design.

## Verify

```bash
curl -s http://<public-ip>/api/health      # -> {"status":"ok",...}
curl -s http://<public-ip>/api/evaluations/benchmark   # metrics table JSON
# open http://<public-ip>/ in a browser, upload a short mp4, watch the pipeline
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| Page won't load | Firewall rule for port 80 missing (Part A step 3), or instance still booting. |
| `DJANGO_SECRET_KEY` error on build | `backend/.env` missing on the VM — rerun the bootstrap script. |
| Upload rejected | Keep videos under 50 MB (`MAX_UPLOAD_SIZE_BYTES` in `.env`). |
| Slow first request | The VM is CPU-only (Always Free has no GPU) — the reference models are CPU-friendly. |
| VM seems unreachable after a while | Free ARM instances can be reclaimed if idle for 7 days; reboot or keep a cron healthcheck. |

## Keeping it alive

Oracle may reclaim Always Free instances idle for 7+ days. A 5-line cron job
that pings `/api/health` every hour keeps it warm:

```bash
echo "* * * * * curl -fsS http://localhost/api/health >/dev/null 2>&1" | crontab -
```