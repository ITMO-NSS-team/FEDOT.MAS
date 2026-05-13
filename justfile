bifrost_port := "9090"
bifrost_url := "http://localhost:" + bifrost_port
searxng_port := "18888"
searxng_dir := env("SEARXNG_DIR", if os() == "macos" { "~/Library/Application Support/fedotmas/searxng" } else if os() == "windows" { "~/AppData/Local/fedotmas/searxng" } else { "~/.local/share/fedotmas/searxng" })

# User section:

venv:
    uv sync
    cp -n .env.example .env 2>/dev/null || true

venv-gaia:
    uv sync --group gaia
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

    mkdir -p "$dir"

    # Stop an older compose project before rewriting docker-compose.yml.
    if [ -f "$dir/docker-compose.yml" ]; then
        docker compose -f "$dir/docker-compose.yml" down --remove-orphans 2>/dev/null || true
    fi

    # Remove old named containers from previous broken/manual installs.
    docker rm -f searxng-core searxng-valkey searxng 2>/dev/null || true

    # Fix ownership problems caused by previous sudo/docker writes.
    if [ -d "$dir" ]; then
        if ! touch "$dir/.fedotmas-write-test" 2>/dev/null; then
            echo "Fixing ownership for $dir"
            sudo chown -R "$USER:$USER" "$dir"
        else
            rm -f "$dir/.fedotmas-write-test"
        fi
    fi

    mkdir -p "$dir/core-config"

    secret_file="$dir/.searxng_secret"
    if [ ! -f "$secret_file" ]; then
        if command -v openssl >/dev/null 2>&1; then
            openssl rand -hex 32 > "$secret_file"
        else
            python3 - <<'PY' > "$secret_file"
import secrets
print(secrets.token_hex(32))
PY
        fi
    fi

    secret="$(cat "$secret_file")"

    cat > "$dir/core-config/settings.yml" <<EOF
use_default_settings: true

general:
  debug: false
  instance_name: "FEDOT.MAS SearXNG"

search:
  safe_search: 0
  autocomplete: ""
  formats:
    - html
    - json

server:
  secret_key: "$secret"
  limiter: false
  image_proxy: false
  method: "GET"

engines:
  - name: wikidata
    disabled: true
  - name: ahmia
    disabled: true
  - name: torch
    disabled: true
EOF

    # If a broken limiter.toml exists from previous attempts, remove it.
    # Limiter is disabled through server.limiter=false in settings.yml.
    rm -f "$dir/core-config/limiter.toml"

    cat > "$dir/docker-compose.yml" <<EOF
services:
  searxng:
    image: searxng/searxng:latest
    container_name: searxng-core
    restart: unless-stopped
    ports:
      - "{{ searxng_host }}:{{ searxng_port }}:8080"
    volumes:
      - ./core-config:/etc/searxng
      - core-data:/var/cache/searxng
    environment:
      - SEARXNG_SECRET=$secret

volumes:
  core-data:
EOF

    echo "SearXNG installed at $dir"
    echo "JSON API will be available at: http://localhost:{{ searxng_port }}/search?q=test&format=json"

searxng-start:
    #!/usr/bin/env bash
    set -euo pipefail

    dir="{{ searxng_dir }}"
    dir="${dir/#\~/$HOME}"

    if [ ! -f "$dir/docker-compose.yml" ] || [ ! -f "$dir/core-config/settings.yml" ]; then
        just searxng-install
    fi

    cd "$dir"

    docker compose up -d --force-recreate

    echo "SearXNG running at http://localhost:{{ searxng_port }}"
    echo "Checking JSON API..."

    for i in $(seq 1 30); do
        if curl -fsS "http://localhost:{{ searxng_port }}/search?q=test&format=json" | python3 -m json.tool >/dev/null 2>&1; then
            echo "SearXNG JSON API OK"
            exit 0
        fi
        sleep 1
    done

    echo "ERROR: SearXNG started, but JSON API check failed."
    echo ""
    echo "Recent logs:"
    docker compose logs --tail 160 searxng || docker logs searxng-core --tail 160
    exit 1

searxng-restart:
    #!/usr/bin/env bash
    set -euo pipefail

    just searxng-stop
    just searxng-start

searxng-reinstall:
    #!/usr/bin/env bash
    set -euo pipefail

    dir="{{ searxng_dir }}"
    dir="${dir/#\~/$HOME}"

    if [ -d "$dir" ] && [ -f "$dir/docker-compose.yml" ]; then
        docker compose -f "$dir/docker-compose.yml" down --remove-orphans 2>/dev/null || true
    fi

    docker rm -f searxng-core searxng-valkey searxng 2>/dev/null || true

    rm -rf "$dir"

    just searxng-install
    just searxng-start

searxng-stop:
    #!/usr/bin/env bash
    set -euo pipefail

    dir="{{ searxng_dir }}"
    dir="${dir/#\~/$HOME}"

    if [ ! -d "$dir" ] || [ ! -f "$dir/docker-compose.yml" ]; then
        docker rm -f searxng-core searxng-valkey searxng 2>/dev/null || true
        echo "SearXNG is not installed at $dir"
        exit 0
    fi

    cd "$dir"
    docker compose down --remove-orphans

searxng-status:
    #!/usr/bin/env bash
    set -euo pipefail

    dir="{{ searxng_dir }}"
    dir="${dir/#\~/$HOME}"

    if [ ! -f "$dir/docker-compose.yml" ]; then
        echo "SearXNG is not installed at $dir"
        exit 1
    fi

    cd "$dir"
    docker compose ps

searxng-logs:
    #!/usr/bin/env bash
    set -euo pipefail

    dir="{{ searxng_dir }}"
    dir="${dir/#\~/$HOME}"

    if [ -f "$dir/docker-compose.yml" ]; then
        cd "$dir"
        docker compose logs -f --tail 200 searxng
    else
        docker logs -f --tail 200 searxng-core
    fi

searxng-check:
    #!/usr/bin/env bash
    set -euo pipefail

    url="http://localhost:{{ searxng_port }}/search?q=test&format=json"

    echo "Checking $url"

    response="$(curl -fsS "$url")"
    echo "$response" | python3 -m json.tool >/dev/null

    echo "SearXNG JSON API OK"

# Lightpanda browser (actually a scraper)

lightpanda-install:
    curl -fsSL https://pkg.lightpanda.io/install.sh | bash

lightpanda-check:
    @lightpanda --version || echo "Lightpanda not installed. Run: just lightpanda-install"

# browser-use (browser automation)

browser-use-install:
    uvx browser-use install

browser-use-check:
    @uvx --from 'browser-use[cli]' browser-use --help > /dev/null 2>&1 && echo "browser-use OK" || echo "browser-use not installed. Run: just browser-use-install"

# GAIA Benchmark

# Run any GAIA script with custom args (e.g., just gaia-run run_gaia --difficulty 1 --split "validation[:5]")
gaia-run script *args:
    #!/usr/bin/env bash
    set -euo pipefail
    SCRIPT_PATH="examples/gaia/{{script}}.py"
    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "Error: script not found: $SCRIPT_PATH"
        echo "Available scripts:"
        ls -1 examples/gaia/run_*.py | xargs -n1 basename | sed 's/\.py$//' | sed 's/^/  - /'
        exit 1
    fi
    echo "Running $SCRIPT_PATH {{args}}"
    uv run python "$SCRIPT_PATH" {{args}}

# Run a batch of 10 questions (batch_num: 0-16 for all 165 validation questions)
gaia-batch batch_num difficulty="all":
    #!/usr/bin/env bash
    set -euo pipefail
    START=$(( {{batch_num}} * 10 ))
    END=$(( START + 10 ))
    SPLIT="validation[$START:$END]"
    echo "Running GAIA batch {{batch_num}}: questions $START-$((END-1)), difficulty={{difficulty}}"
    just gaia-run run_gaia --difficulty {{difficulty}} --split "$SPLIT"

# Run all 17 batches (165 validation questions total)
gaia-all difficulty="all":
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running all 17 batches (165 questions total), difficulty={{difficulty}}"
    for i in $(seq 0 16); do
        echo "========================================="
        echo "Batch $i/16"
        echo "========================================="
        just gaia-batch $i {{difficulty}}
        echo ""
    done
    echo "All batches completed!"

# Run a range of batches (e.g., just gaia-range 0 4)
gaia-range start_batch end_batch difficulty="all":
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running batches {{start_batch}}-{{end_batch}}, difficulty={{difficulty}}"
    for i in $(seq {{start_batch}} {{end_batch}}); do
        echo "========================================="
        echo "Batch $i/{{end_batch}}"
        echo "========================================="
        just gaia-batch $i {{difficulty}}
        echo ""
    done
    echo "Batches {{start_batch}}-{{end_batch}} completed!"

# Resume from a specific batch to the end
gaia-resume from_batch difficulty="all":
    @just gaia-range {{from_batch}} 16 {{difficulty}}
