# searxng-search MCP server

Web search via self-hosted SearXNG. Registered automatically in FEDOT.MAS.


### Set up SearXNG

SearXNG provides a self-hosted metasearch engine for web searches without API rate limits.

**Prerequisites:** Docker and Docker Compose installed

**Manual setup (without just):**

```bash
# Clone SearXNG Docker repository
cd ~
git clone https://github.com/searxng/searxng-docker.git
cd searxng-docker

# Generate secret key
sed -i "s|ultrasecretkey|$(openssl rand -hex 32)|g" searxng/settings.yml

# Configure search formats
cat >> searxng/settings.yml << EOF
search:
  formats:
    - html
    - json
    - csv
    - rss
EOF

# Update port in docker-compose.yaml
sed -i "s/127.0.0.1:8080:8080/127.0.0.1:8888:8080/" docker-compose.yaml

# Start services
docker compose up -d

# Verify running
curl http://localhost:8888
```

**Important:** The `search.formats` configuration is required for API access. Without it, JSON responses won't be available.

SearXNG will be available at `http://localhost:8888`


## Usage in FEDOT.MAS

In auto mode the meta-agent assigns the tool automatically when the task requires web search:

```python
mas = MAS()
state = await mas.run("What are the latest AI research papers from 2025?")
```

To assign it explicitly to an agent:

```python
AgentConfig(
    name="researcher",
    instruction="Search the web and answer: {user_query}",
    output_key="result",
    tools=["searxng-search"],
)
```
