#!/usr/bin/env bash
# Bootstrap a fresh Ubuntu 22.04/24.04 VM (e.g. an Azure VM) to run the app.
# Run once, as the default sudo-capable user (e.g. 'azureuser'):
#   curl -fsSL <raw-url>/scripts/deploy/vm-bootstrap.sh | bash
# or scp it over and `bash vm-bootstrap.sh`.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/BenBrahimMazen/arxiv-rag.git}"
APP_DIR="${APP_DIR:-$HOME/arxiv-rag}"

echo "### Updating packages ..."
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl git ufw

echo "### Installing Docker Engine + compose plugin ..."
if ! command -v docker >/dev/null 2>&1; then
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin
  sudo usermod -aG docker "$USER"
fi

echo "### Configuring host firewall (ufw) ..."
sudo ufw allow OpenSSH || true
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8000/tcp
sudo ufw allow 8501/tcp
sudo ufw --force enable

echo "### Cloning repo into $APP_DIR ..."
if [ ! -d "$APP_DIR/.git" ]; then
  git clone "$REPO_URL" "$APP_DIR"
fi

cd "$APP_DIR"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "!!! Edit $APP_DIR/.env with your secrets, DOMAIN and CERTBOT_EMAIL before deploying."
fi

cat <<'EOF'

Next steps:
  1) Log out and back in (so the docker group applies), or run: newgrp docker
  2) Edit .env  (secrets, DOMAIN, CERTBOT_EMAIL)
  3) In the Azure portal, open ports 80, 443, 8000, 8501 in the VM's Network Security Group
  4) Point your domain's DNS A record at this VM's static public IP
  5) bash scripts/deploy/init-letsencrypt.sh
  6) docker compose -f docker-compose.prod.yml up -d --build
  7) Ingest data: docker compose -f docker-compose.prod.yml exec api python -m scripts.ingest --max-papers 50
EOF
