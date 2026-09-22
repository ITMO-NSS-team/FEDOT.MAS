# Rubber recipe predictor MCP server

Exposes `predict_rubber_recipe`, a deterministic research tool backed by the
20-row open SBR/NR/N220 dataset in `experiments/rubber_recipe_mas/open_data`.
FEDOT.MAS discovers the server under the name `rubber-recipe-predictor`.

The tool returns one recipe, four property predictions, LOOCV metrics, domain
distance, and data provenance. It is an interpolation demonstrator, not a tire
safety or production recommendation.
