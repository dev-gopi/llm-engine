#!/usr/bin/env sh
set -eu

certificate_dir="${1:-deploy/certs}"
mkdir -p "$certificate_dir"

openssl req -x509 -newkey rsa:3072 -sha256 -nodes \
  -keyout "$certificate_dir/server.key" \
  -out "$certificate_dir/server.crt" \
  -days 30 \
  -subj "/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"

chmod 600 "$certificate_dir/server.key"
printf 'Created development-only certificate in %s\n' "$certificate_dir"
printf 'Use a trusted CA certificate for public deployment.\n'
