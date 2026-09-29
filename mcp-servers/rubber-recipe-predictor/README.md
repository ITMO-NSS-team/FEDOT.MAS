# Rubber property predictor MCP server

Exposes `predict_rubber_properties`, a deterministic research tool backed by the
20-row open SBR/NR/N220 dataset in `experiments/rubber_recipe_mas/open_data`.
FEDOT.MAS discovers the server under the name `rubber-recipe-predictor`.

Supply all 11 recipe components in phr. The tool preserves the supplied recipe
and returns four property predictions, LOOCV metrics, domain distance and data
provenance. It does not generate or optimize recipes. Predictions require
laboratory validation before production use.
