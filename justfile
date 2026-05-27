bifrost_port := "9090"
bifrost_url := "http://localhost:" + bifrost_port
searxng_port := "18888"
searxng_host := env("SEARXNG_HOST", "127.0.0.1")
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

    if [ -f "$dir/docker-compose.yml" ]; then
        docker compose -f "$dir/docker-compose.yml" down --remove-orphans 2>/dev/null || true
    fi

    docker rm -f searxng-core searxng-valkey searxng 2>/dev/null || true

    if ! touch "$dir/.fedotmas-write-test" 2>/dev/null; then
        echo "Fixing ownership for $dir"
        sudo chown -R "$USER:$USER" "$dir"
    else
        rm -f "$dir/.fedotmas-write-test"
    fi

    mkdir -p "$dir/core-config"

    secret_file="$dir/.searxng_secret"

    if [ ! -f "$secret_file" ]; then
        if command -v openssl >/dev/null 2>&1; then
            openssl rand -hex 32 > "$secret_file"
        elif command -v python3 >/dev/null 2>&1; then
            python3 -c 'import secrets; print(secrets.token_hex(32))' > "$secret_file"
        else
            date +%s | sha256sum | awk '{print $1}' > "$secret_file"
        fi
    fi

    secret="$(cat "$secret_file")"

    {
        printf '%s\n' 'use_default_settings: true'
        printf '%s\n' ''
        printf '%s\n' 'general:'
        printf '%s\n' '  debug: false'
        printf '%s\n' '  instance_name: "FEDOT.MAS SearXNG"'
        printf '%s\n' ''
        printf '%s\n' 'search:'
        printf '%s\n' '  safe_search: 0'
        printf '%s\n' '  autocomplete: ""'
        printf '%s\n' '  formats:'
        printf '%s\n' '    - html'
        printf '%s\n' '    - json'
        printf '%s\n' ''
        printf '%s\n' 'server:'
        printf '%s\n' "  secret_key: \"$secret\""
        printf '%s\n' '  limiter: false'
        printf '%s\n' '  image_proxy: false'
        printf '%s\n' '  method: "GET"'
        printf '%s\n' ''
        printf '%s\n' 'engines:'
        printf '%s\n' '  - name: wikidata'
        printf '%s\n' '    disabled: true'
        printf '%s\n' '  - name: ahmia'
        printf '%s\n' '    disabled: true'
        printf '%s\n' '  - name: torch'
        printf '%s\n' '    disabled: true'
        printf '%s\n' '  - name: google'
        printf '%s\n' '    disabled: true'
        printf '%s\n' '  - name: brave'
        printf '%s\n' '    disabled: true'
    } > "$dir/core-config/settings.yml"

    rm -f "$dir/core-config/limiter.toml"

    {
        printf '%s\n' 'services:'
        printf '%s\n' '  searxng:'
        printf '%s\n' '    image: searxng/searxng:latest'
        printf '%s\n' '    container_name: searxng-core'
        printf '%s\n' '    restart: unless-stopped'
        printf '%s\n' '    ports:'
        printf '%s\n' '      - "{{ searxng_host }}:{{ searxng_port }}:8080"'
        printf '%s\n' '    volumes:'
        printf '%s\n' '      - ./core-config:/etc/searxng'
        printf '%s\n' '      - core-data:/var/cache/searxng'
        printf '%s\n' ''
        printf '%s\n' 'volumes:'
        printf '%s\n' '  core-data:'
    } > "$dir/docker-compose.yml"

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

    if [ -n "$dir" ] && [ "$dir" != "/" ] && [ "$dir" != "$HOME" ]; then
        if [ -d "$dir" ]; then
            sudo chown -R "$USER:$USER" "$dir" 2>/dev/null || true
            chmod -R u+rwX "$dir" 2>/dev/null || true
        fi
        rm -rf "$dir"
    else
        echo "ERROR: unsafe searxng_dir: $dir"
        exit 1
    fi

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

    if command -v python3 >/dev/null 2>&1; then
        response="$(curl -fsS "$url")"
        echo "$response" | python3 -m json.tool >/dev/null
    else
        curl -fsS "$url" >/dev/null
    fi

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
    SCRIPT_PATH="benchmarks/gaia/{{script}}.py"
    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "Error: script not found: $SCRIPT_PATH"
        echo "Available scripts:"
        ls -1 benchmarks/gaia/run_*.py | xargs -n1 basename | sed 's/\.py$//' | sed 's/^/  - /'
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
