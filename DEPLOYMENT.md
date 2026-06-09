# 🚀 Deployment runbook (AWS EC2 + nginx + HTTPS + auto-deploy)

This guide takes the app from a fresh AWS account to a public **HTTPS** URL that
**auto-deploys on every push to `main`**. Everything here is one-paste/one-command;
the steps that need your AWS account or a domain are called out.

> Prefer GCP Cloud Run? See [the Cloud Run notes](#alternative-gcp-cloud-run) at the bottom.

---

## Architecture in production

```
Internet ──► nginx (80/443, TLS)
                ├── /        ─► frontend (Streamlit :8501)
                └── /api/    ─► api (FastAPI :8000)  ── Postgres :5432
                                                     └─ Chroma (volume)
certbot ──► renews Let's Encrypt certs every 12h
```

Files involved:
- [docker-compose.prod.yml](docker-compose.prod.yml) — postgres + api + frontend + nginx + certbot
- [nginx/app.conf.template](nginx/app.conf.template) — reverse proxy (SSE-aware, websocket upgrade)
- [scripts/deploy/ec2-bootstrap.sh](scripts/deploy/ec2-bootstrap.sh) — installs Docker + firewall on the instance
- [scripts/deploy/init-letsencrypt.sh](scripts/deploy/init-letsencrypt.sh) — obtains the first certificate
- [.github/workflows/deploy.yml](.github/workflows/deploy.yml) — SSH auto-deploy on push to main

---

## 1. Launch the EC2 instance  *(AWS account — manual)*

1. **EC2 → Launch instance**
   - AMI: **Ubuntu Server 24.04 LTS**
   - Type: **t3.micro** (or t2.micro) — free-tier eligible
   - Key pair: create/select one; **save the `.pem`** (this is your `EC2_SSH_KEY`)
   - Storage: bump to **30 GB** (ML wheels + models are large)

2. **Security group** — inbound rules:

   | Type        | Port | Source     | Why |
   | ----------- | ---- | ---------- | --- |
   | SSH         | 22   | _your IP_  | admin access |
   | HTTP        | 80   | 0.0.0.0/0  | ACME challenge + redirect |
   | HTTPS       | 443  | 0.0.0.0/0  | public traffic |
   | Custom TCP  | 8000 | 0.0.0.0/0  | direct API (optional/debug) |
   | Custom TCP  | 8501 | 0.0.0.0/0  | direct frontend (optional/debug) |

3. **Elastic IP** (stable IP across reboots):
   - **EC2 → Elastic IPs → Allocate**, then **Associate** it with the instance.
   - Note the address — call it `<EIP>`.

---

## 2. Point DNS at the server  *(domain registrar — manual)*

Create an **A record**: `arxiv-rag.example.com  →  <EIP>`.
Verify: `dig +short arxiv-rag.example.com` returns `<EIP>`.

> No domain? You can still run over plain HTTP on `http://<EIP>:8501` (frontend)
> and `http://<EIP>:8000` (API). HTTPS/certbot requires a real domain.

---

## 3. Bootstrap the instance

SSH in and run the bootstrap (installs Docker, compose, ufw, clones the repo):

```bash
ssh -i your-key.pem ubuntu@<EIP>
export REPO_URL=https://github.com/your-username/arxiv-rag.git
curl -fsSL "$REPO_URL/raw/main/scripts/deploy/ec2-bootstrap.sh" | bash
newgrp docker            # apply the docker group without re-login
```

---

## 4. Configure secrets on the server  *(never commit these)*

```bash
cd ~/arxiv-rag
cp .env.example .env
nano .env
```

Set at minimum:
- `DOMAIN=arxiv-rag.example.com`
- `CERTBOT_EMAIL=you@example.com`
- `POSTGRES_PASSWORD=<a strong password>`
- `OPENAI_API_KEY=...` (and `LLM_BACKEND=openai`) — or keep the free local defaults
- optionally `COHERE_API_KEY`, `PINECONE_API_KEY`, etc.

> The `.env` lives **only on the server**. It is git-ignored and never pushed.

---

## 5. Obtain the TLS certificate (one-time)

```bash
bash scripts/deploy/init-letsencrypt.sh
```

This serves the HTTP-01 challenge through nginx and installs a real Let's Encrypt
certificate. Use `STAGING=1 bash scripts/deploy/init-letsencrypt.sh` first if you
want to avoid rate limits while testing.

---

## 6. Start the stack & ingest data

```bash
docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml exec api python -m scripts.ingest --max-papers 50
```

Visit **https://arxiv-rag.example.com** (frontend) and
**https://arxiv-rag.example.com/api/docs** (API).

---

## 7. Enable auto-deploy on push to main

Add these **GitHub repository secrets** (Settings → Secrets and variables → Actions):

| Secret         | Value |
| -------------- | ----- |
| `EC2_HOST`     | `<EIP>` or your domain |
| `EC2_USER`     | `ubuntu` |
| `EC2_SSH_KEY`  | contents of your `.pem` private key |
| `EC2_APP_DIR`  | `/home/ubuntu/arxiv-rag` |

Now every push to `main` runs [deploy.yml](.github/workflows/deploy.yml): it SSHes in,
`git reset --hard origin/main`, rebuilds, restarts, and health-checks the API.

---

## Operations cheatsheet

```bash
docker compose -f docker-compose.prod.yml ps             # status
docker compose -f docker-compose.prod.yml logs -f api    # tail API logs
docker compose -f docker-compose.prod.yml restart nginx  # reload proxy
docker compose -f docker-compose.prod.yml down           # stop everything
curl -fsS https://arxiv-rag.example.com/api/health       # health check
```

Cert renewal is automatic (certbot container, twice daily; nginx reloads every 6h).
Test renewal: `docker compose -f docker-compose.prod.yml run --rm certbot renew --dry-run`.

---

## Alternative: GCP Cloud Run

Cloud Run is serverless (no nginx/EIP/certbot needed — TLS and a public URL are
provided automatically), but it runs **one container per service** and has no
persistent disk, so use **Pinecone** (not Chroma) and **managed Postgres** (Cloud SQL).

```bash
# Build & push the API image
gcloud builds submit --tag gcr.io/$PROJECT/arxiv-rag-api -f Dockerfile.api .
gcloud run deploy arxiv-rag-api \
  --image gcr.io/$PROJECT/arxiv-rag-api \
  --region us-central1 --allow-unauthenticated \
  --set-env-vars VECTOR_BACKEND=pinecone,EMBEDDING_BACKEND=openai \
  --set-secrets OPENAI_API_KEY=openai-key:latest,PINECONE_API_KEY=pinecone-key:latest

# Frontend service (point API_URL at the API service URL)
gcloud builds submit --tag gcr.io/$PROJECT/arxiv-rag-frontend -f Dockerfile.frontend .
gcloud run deploy arxiv-rag-frontend \
  --image gcr.io/$PROJECT/arxiv-rag-frontend \
  --region us-central1 --allow-unauthenticated \
  --set-env-vars API_URL=https://arxiv-rag-api-xxxx.run.app
```
