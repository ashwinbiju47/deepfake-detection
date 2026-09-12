#!/usr/bin/env bash
# Oracle Cloud Always Free bootstrap — one-command deploy of the Deepfake
# Detection Platform on a fresh Ubuntu 22.04/24.04 ARM or x86 VM.
#
# Usage (on the VM, as a non-root sudo user):
#   bash oracle_bootstrap.sh [repo-url] [branch]
# Defaults:
#   repo-url = https://github.com/ashwinbiju47/deepfake-detection.git
#   branch   = freebuff/need-to-change-4dac48fb-bf8f-4f32-800f-d43831556497
#
# What it does:
#   1. Installs Docker Engine + compose plugin (Ubuntu apt).
#   2. Clones the repo (or reuses the current directory if it already has a
#      docker-compose.yml).
#   3. Writes backend/.env with a freshly generated DJANGO_SECRET_KEY.
#   4. docker compose up -d --build  (first build pulls ~2-3 GB of wheels:
#      CPU PyTorch + TensorFlow + OpenCV + MediaPipe + librosa — allow 15-25 min).
#   5. Prints the public URL and status.
set -euo pipefail

REPO_URL="${1:-https://github.com/ashwinbiju47/deepfake-detection.git}"
BRANCH="${2:-freebuff/need-to-change-4dac48fb-bf8f-4f32-800f-d43831556497}"
APP_DIR="${APP_DIR:-$HOME/deepfake-detection}"

echo "==> [1/5] Installing Docker Engine + compose plugin"
if ! command -v docker >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    # shellcheck disable=SC1091
    . /etc/os-release
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" |
        sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    sudo usermod -aG docker "$USER"
    echo "    (log out/in once if 'docker' needs sudo in this session)"
fi
docker --version
docker compose version

echo "==> [2/5] Fetching application code"
if [ -f "$APP_DIR/docker-compose.yml" ]; then
    echo "    reusing existing checkout at $APP_DIR"
else
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"

echo "==> [3/5] Writing backend/.env"
if [ ! -f backend/.env ]; then
    SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(64))" 2>/dev/null || openssl rand -base64 48)
    cat > backend/.env <<EOF
DJANGO_SECRET_KEY=$SECRET_KEY
DJANGO_DEBUG=false
DJANGO_ALLOWED_HOSTS=*
DJANGO_DB_ENGINE=postgres
POSTGRES_DB=deepfake
POSTGRES_USER=deepfake
POSTGRES_PASSWORD=$(python3 -c "import secrets; print(secrets.token_urlsafe(24))" 2>/dev/null || openssl rand -base64 24)
DJANGO_SERVE_SPA=true
EXTERNAL_URL_ENABLED=true
HEATMAP_ENABLED=true
REPORT_ENABLED=true
CELERY_WORKER_CONCURRENCY=4
EOF
    chmod 600 backend/.env
    echo "    wrote backend/.env (secrets generated locally)"
else
    echo "    backend/.env already exists — leaving it untouched"
fi

echo "==> [4/5] Building and starting (first build takes 15-25 min)"
sudo docker compose up -d --build

echo "==> [5/5] Status"
sleep 5
sudo docker compose ps
echo
IP=$(curl -s -4 ifconfig.me || echo "<public-ip>")
echo "Application should be live at:  http://$IP/"
echo "Health check:                   curl -s http://$IP/api/health"
echo
echo "Logs:      sudo docker compose logs -f web"
echo "Rebuild:   cd $APP_DIR && sudo docker compose up -d --build"
echo "Stop:      cd $APP_DIR && sudo docker compose down"