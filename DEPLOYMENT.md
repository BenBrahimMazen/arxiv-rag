# 🚀 Deployment runbook (Azure VM + nginx + HTTPS + auto-deploy)

This guide takes the app from a fresh Azure VM to a public **HTTPS** URL that
**auto-deploys on every push to `main`**. It is designed to run on the **Azure for
Students** credit included in the GitHub Student Developer Pack (no credit card required).
Every step is one-paste/one-command; the steps that need the Azure portal or a domain are
called out.

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
- [scripts/deploy/vm-bootstrap.sh](scripts/deploy/vm-bootstrap.sh) — installs Docker + firewall on the VM
- [scripts/deploy/init-letsencrypt.sh](scripts/deploy/init-letsencrypt.sh) — obtains the first certificate
- [.github/workflows/deploy.yml](.github/workflows/deploy.yml) — SSH auto-deploy on push to main

---

## 1. Create the Azure VM  *(Azure portal — manual)*

1. **Azure Portal → Virtual machines → Create → Azure virtual machine**
   - Image: **Ubuntu Server 24.04 LTS**
   - Size: **Standard B2s** (2 vCPU, 4 GB RAM) — the 4 GB comfortably builds the ML image;
     B1s (1 GB) is too small for the torch build.
   - Authentication: **SSH public key**; download/save the `.pem` (this is your `VM_SSH_KEY`)
   - Username: e.g. **azureuser**
   - Disk: **30 GB** standard SSD (ML wheels + models are large)

2. **Networking — open the required inbound ports** in the VM's
   **Network Security Group** (Networking → Add inbound port rule):

   | Port | Source     | Why |
   | ---- | ---------- | --- |
   | 22   | _your IP_  | SSH admin access |
   | 80   | Any        | ACME challenge + HTTP→HTTPS redirect |
   | 443  | Any        | public HTTPS traffic |
   | 8000 | Any        | direct API (optional/debug) |
   | 8501 | Any        | direct frontend (optional/debug) |

3. **Static public IP** (so the address survives reboots):
   - VM → **Networking → Network interface → IP configurations → ipconfig1**
   - Set **Public IP → Association: Static**, save. Note the address — call it `<IP>`.

> Cost tip: a B2s runs ~$30/month if left on 24/7; your $100 student credit covers ~3 months.
> **Stop (deallocate) the VM** from the portal when you are not demoing to make the credit last.

---

## 2. Point DNS at the server  *(domain registrar — manual)*

Create an **A record**: `arxiv-rag.example.com  →  <IP>`.
Verify: `dig +short arxiv-rag.example.com` returns `<IP>`.

> No domain? You can still run over plain HTTP on `http://<IP>:8501` (frontend) and
> `http://<IP>:8000` (API). HTTPS/certbot requires a real domain. Free options include a
> `*.nip.io` hostname or a free subdomain from DuckDNS/Freenom-style providers.

---

## 3. Bootstrap the VM

SSH in and run the bootstrap (installs Docker, compose, ufw, clones the repo):

```bash
ssh -i your-key.pem azureuser@<IP>
curl -fsSL https://raw.githubusercontent.com/BenBrahimMazen/arxiv-rag/main/scripts/deploy/vm-bootstrap.sh | bash
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
- `LLM_BACKEND=groq` and `GROQ_API_KEY=...` (free, from https://console.groq.com/keys)
- keep the free local defaults for `EMBEDDING_BACKEND`, `RERANKER_BACKEND`, `VECTOR_BACKEND`

> The `.env` lives **only on the server**. It is git-ignored and never pushed.

---

## 5. Obtain the TLS certificate (one-time)

```bash
bash scripts/deploy/init-letsencrypt.sh
```

This serves the HTTP-01 challenge through nginx and installs a real Let's Encrypt
certificate. Use `STAGING=1 bash scripts/deploy/init-letsencrypt.sh` first if you want to
avoid rate limits while testing.

---

## 6. Start the stack & ingest data

The images are **prebuilt in CI and published to GHCR** (the
[Build Images](.github/workflows/build-images.yml) workflow), so the VM only pulls them —
no torch-heavy build on the VM:

```bash
docker compose -f docker-compose.prod.yml pull api frontend
docker compose -f docker-compose.prod.yml up -d
docker compose -f docker-compose.prod.yml exec api python -m scripts.ingest --max-papers 50
```

Visit **https://arxiv-rag.example.com** (frontend) and
**https://arxiv-rag.example.com/api/docs** (API).

> **One-time:** after the first `Build Images` run, make the two GHCR packages **public**
> (GitHub → your profile → Packages → `arxiv-rag-api` / `arxiv-rag-frontend` → Package
> settings → Change visibility → Public) so the VM can pull them without authenticating.
> To keep them private instead, run `docker login ghcr.io` on the VM with a PAT that has
> `read:packages`.

---

## 7. Enable auto-deploy on push to main

Add these **GitHub repository secrets** (Settings → Secrets and variables → Actions):

| Secret        | Value |
| ------------- | ----- |
| `VM_HOST`     | `<IP>` or your domain |
| `VM_USER`     | `azureuser` |
| `VM_SSH_KEY`  | contents of your `.pem` private key |
| `VM_APP_DIR`  | `/home/azureuser/arxiv-rag` |

Now every push to `main` runs [deploy.yml](.github/workflows/deploy.yml): it SSHes in,
`git reset --hard origin/main`, rebuilds, restarts, and health-checks the API. (Until these
secrets are set, the deploy job simply no-ops, so it never fails.)

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

## Optional: build images in CI instead of on the VM

A small VM can struggle to build the torch-heavy API image. To avoid building on the server,
build the images in GitHub Actions, push them to the GitHub Container Registry (GHCR), and
have the VM only `docker compose pull`. Ask for the `build-and-push` workflow if you want
this — it keeps the VM lightweight and deploys in seconds.
