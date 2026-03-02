# fedotmas

Core library for FEDOT.MAS.

## Bifrost

Run it locally via Docker:

```
just bifrost
```

And use it as .env variable:

```
FEDOTMAS_DEFAULT_PROXY=bifrost
```

Bifrost will be available at `http://localhost:9090` by default (to switch default url use var: `FEDOTMAS_BIFROST_BASE_URL=`). To stop:

```
just bifrost-stop
```

## Usage

```python
import asyncio
from fedotmas import MAS

async def main():
    mas = MAS()
    state = await mas.run("Compare Python and Rust for CLI tools")
    print(state)

asyncio.run(main())
```
