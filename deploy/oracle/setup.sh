#!/usr/bin/env bash
# Install (or update) the Gradio demo on an Ubuntu 22.04/24.04 server, e.g. an
# Oracle Cloud Always Free VM. Safe to re-run: it pulls the latest code and restarts.
#
#   curl -fsSL https://raw.githubusercontent.com/venkateshmedasani/SLM-based-hate-speech-recognition/main/deploy/oracle/setup.sh | bash
#   curl -fsSL .../setup.sh | bash -s -- demo.example.com     # with HTTPS for a domain you own
#
# Result: demo on http://<server-ip>/ (or https://<domain>/), auto-restarting systemd service.

set -euo pipefail

DOMAIN="${1:-}"
REPO_URL="https://github.com/venkateshmedasani/SLM-based-hate-speech-recognition.git"
APP_DIR="/opt/slm-demo"
APP_USER="slmdemo"
SERVICE="slm-demo"
export DEBIAN_FRONTEND=noninteractive

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

log "Installing system packages"
sudo apt-get update -y
sudo apt-get install -y git curl gpg python3 python3-venv python3-pip iptables-persistent
if ! sudo apt-get install -y caddy; then
  # Ubuntu 22.04 has no caddy package; use Caddy's official apt repository.
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key | sudo gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y caddy
fi

# Small shapes (e.g. 1 OCPU / 6 GB) need swap: the model takes ~5 GB of RAM in float32.
mem_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
if [ "$mem_kb" -lt 12000000 ] && ! swapon --show | grep -q '/swapfile'; then
  log "Adding 8 GB swap (RAM is $((mem_kb / 1024 / 1024)) GB)"
  sudo fallocate -l 8G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

log "Creating service user and fetching code"
id -u "$APP_USER" >/dev/null 2>&1 || sudo useradd --system --create-home --home-dir "$APP_DIR" --shell /usr/sbin/nologin "$APP_USER"
if [ -d "$APP_DIR/repo/.git" ]; then
  sudo -u "$APP_USER" git -C "$APP_DIR/repo" pull --ff-only
else
  sudo -u "$APP_USER" git clone --depth 1 "$REPO_URL" "$APP_DIR/repo"
fi

log "Installing Python dependencies (CPU PyTorch, a few minutes)"
sudo -u "$APP_USER" python3 -m venv "$APP_DIR/venv"
sudo -u "$APP_USER" "$APP_DIR/venv/bin/pip" install --quiet --upgrade pip
sudo -u "$APP_USER" "$APP_DIR/venv/bin/pip" install --quiet -r "$APP_DIR/repo/demo/requirements.txt"

log "Downloading the model (~2.3 GB, one-off)"
sudo -u "$APP_USER" env HF_HOME="$APP_DIR/hf-cache" "$APP_DIR/venv/bin/python" -c \
  "import sys; sys.path.insert(0, '$APP_DIR/repo/demo'); import app; print(app.classify('have a great day')[0])"

log "Installing systemd service"
sudo tee /etc/systemd/system/$SERVICE.service >/dev/null <<EOF
[Unit]
Description=SLM hate speech detection demo (Gradio)
After=network-online.target
Wants=network-online.target

[Service]
User=$APP_USER
WorkingDirectory=$APP_DIR/repo
Environment=HF_HOME=$APP_DIR/hf-cache
Environment=GRADIO_SERVER_NAME=127.0.0.1
Environment=GRADIO_SERVER_PORT=7860
Environment=GRADIO_ANALYTICS_ENABLED=False
ExecStart=$APP_DIR/venv/bin/python demo/app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE" >/dev/null
sudo systemctl restart "$SERVICE"

log "Configuring Caddy reverse proxy"
if [ -n "$DOMAIN" ]; then
  site="$DOMAIN"
else
  site=":80"
fi
sudo tee /etc/caddy/Caddyfile >/dev/null <<EOF
$site {
	encode gzip
	reverse_proxy 127.0.0.1:7860
}
EOF
sudo systemctl enable caddy >/dev/null
sudo systemctl restart caddy

log "Opening ports 80/443 in the VM firewall"
# Oracle's Ubuntu images ship iptables rules that REJECT everything except SSH.
for port in 80 443; do
  sudo iptables -C INPUT -p tcp --dport "$port" -m state --state NEW -j ACCEPT 2>/dev/null \
    || sudo iptables -I INPUT 1 -p tcp --dport "$port" -m state --state NEW -j ACCEPT
done
sudo netfilter-persistent save >/dev/null

log "Waiting for the app to come up"
for _ in $(seq 1 60); do
  if curl -fs -o /dev/null http://127.0.0.1:7860/; then break; fi
  sleep 3
done
if curl -fs -o /dev/null http://127.0.0.1:7860/; then
  ip=$(curl -fs https://ifconfig.me || echo "<server-ip>")
  log "Done. Demo is live at: ${DOMAIN:+https://$DOMAIN/ and }http://$ip/"
  echo "Logs:    sudo journalctl -u $SERVICE -f"
  echo "Update:  re-run this script"
else
  echo "App did not respond yet. Check: sudo journalctl -u $SERVICE -n 100" >&2
  exit 1
fi
