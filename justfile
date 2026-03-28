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
