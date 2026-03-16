bifrost_port := "9090"
bifrost_url := "http://localhost:" + bifrost_port
searxng_port := "18888"
searxng_dir := "~/.local/share/fedotmas/searxng"

# User section:

venv:
    uv sync
    cp -n .env.example .env 2>/dev/null || true

# Dev section:

venv-dev:
    uv sync --group dev
    uv run prek install
    @echo "Dev environment ready"

upd-hooks:
    prek uninstall
    prek install

lint:
    uv run ruff check . --fix
    uv run ruff format .

typecheck:
    uv run ty check packages/

check: lint typecheck

test-unit:
    uv run pytest packages/fedotmas/tests/ -v

# bifrost

bifrost:
    docker run -d --name bifrost \
      -p {{ bifrost_port }}:8080 \
      -v bifrost_data:/app/data \
      -v $(pwd)/bifrost/config.json:/app/data/config.json \
      --env-file .env \
      maximhq/bifrost
    @echo "Bifrost running at {{ bifrost_url }}"

bifrost-stop:
    docker stop bifrost && docker rm bifrost

# SearXNG

searxng-install:
    #!/usr/bin/env bash
    set -euo pipefail
    dir="{{ searxng_dir }}"
    dir="${dir/#\~/$HOME}"
    if [ -d "$dir" ]; then
        echo "SearXNG already installed at $dir"
        exit 0
    fi
    mkdir -p "$dir"
    git clone https://github.com/searxng/searxng-docker.git "$dir"
    cd "$dir"
    # Enable JSON output format
    mkdir -p searxng
    cat > searxng/settings.yml <<'SETTINGS'
    use_default_settings: true
    server:
      secret_key: "$(openssl rand -hex 32)"
    search:
      formats:
        - html
        - json
    SETTINGS
    # Set port in .env
    sed -i "s|^SEARXNG_HOSTNAME=.*|SEARXNG_HOSTNAME=localhost|" .env 2>/dev/null || true
    sed -i "s|127.0.0.1:8080|127.0.0.1:{{ searxng_port }}|" docker-compose.yaml 2>/dev/null || true
    echo "SearXNG installed at $dir — run: just searxng-start"

searxng-start:
    #!/usr/bin/env bash
    set -euo pipefail
    dir="{{ searxng_dir }}"
    dir="${dir/#\~/$HOME}"
    cd "$dir"
    docker compose up -d
    echo "SearXNG running at http://localhost:{{ searxng_port }}"

searxng-stop:
    #!/usr/bin/env bash
    set -euo pipefail
    dir="{{ searxng_dir }}"
    dir="${dir/#\~/$HOME}"
    cd "$dir"
    docker compose down

searxng-status:
    #!/usr/bin/env bash
    set -euo pipefail
    dir="{{ searxng_dir }}"
    dir="${dir/#\~/$HOME}"
    cd "$dir"
    docker compose ps
