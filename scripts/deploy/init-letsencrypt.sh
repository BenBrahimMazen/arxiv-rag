#!/usr/bin/env bash
# One-time bootstrap of a Let's Encrypt certificate for ${DOMAIN}.
#
# Prereqs (see DEPLOYMENT.md):
#   - DNS A record for $DOMAIN points at the VM's static public IP
#   - .env contains DOMAIN and CERTBOT_EMAIL
#   - ports 80 and 443 open in the VM's Network Security Group
#
# It starts nginx with a temporary self-signed cert so the HTTP-01 challenge can
# be served, requests the real certificate, then reloads nginx.
set -euo pipefail
cd "$(dirname "$0")/../.."

# Load DOMAIN / CERTBOT_EMAIL from .env
set -a
# shellcheck disable=SC1091
source .env
set +a

: "${DOMAIN:?set DOMAIN in .env}"
: "${CERTBOT_EMAIL:?set CERTBOT_EMAIL in .env}"

STAGING="${STAGING:-0}"   # set STAGING=1 to use Let's Encrypt staging (no rate limits)
cert_path="certbot/conf/live/$DOMAIN"

mkdir -p certbot/conf certbot/www

if [ -d "$cert_path" ]; then
  echo "Certificate for $DOMAIN already exists at $cert_path — nothing to do."
  echo "Delete that directory to force re-issuance."
  exit 0
fi

echo "### Creating a temporary self-signed certificate for $DOMAIN ..."
mkdir -p "$cert_path"
docker run --rm -v "$(pwd)/certbot/conf:/etc/letsencrypt" certbot/certbot \
  sh -c "openssl req -x509 -nodes -newkey rsa:2048 -days 1 \
    -keyout '/etc/letsencrypt/live/$DOMAIN/privkey.pem' \
    -out '/etc/letsencrypt/live/$DOMAIN/fullchain.pem' \
    -subj '/CN=localhost'"

echo "### Starting nginx ..."
docker compose -f docker-compose.prod.yml up -d nginx

echo "### Removing temporary certificate ..."
rm -rf "certbot/conf/live/$DOMAIN" \
       "certbot/conf/archive/$DOMAIN" \
       "certbot/conf/renewal/$DOMAIN.conf"

staging_arg=""
if [ "$STAGING" != "0" ]; then staging_arg="--staging"; fi

echo "### Requesting Let's Encrypt certificate for $DOMAIN ..."
docker compose -f docker-compose.prod.yml run --rm certbot certbot certonly \
  --webroot -w /var/www/certbot \
  $staging_arg \
  --email "$CERTBOT_EMAIL" \
  -d "$DOMAIN" \
  --rsa-key-size 4096 \
  --agree-tos \
  --no-eff-email \
  --force-renewal

echo "### Reloading nginx ..."
docker compose -f docker-compose.prod.yml exec nginx nginx -s reload

echo "Done. HTTPS should now be live at https://$DOMAIN"
