# sandbox-light MCP server

Safe code execution without imports or file access.
Powered by [pydantic-monty](https://github.com/pydantic/monty).

## Tools

- `execute` — run code once, get result
- `repl` — persistent session with shared state between calls (like Jupyter)

## Usage

```python
AgentConfig(tools=["sandbox-light"])
```
