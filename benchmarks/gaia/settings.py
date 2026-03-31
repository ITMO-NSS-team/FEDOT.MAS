from pydantic_settings import BaseSettings, SettingsConfigDict


class GaiaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GAIA_")

    solver_model: str = "openai/gpt-4o-mini"
    difficulty: str = "all"
    split: str = "validation"
    max_iterations: int = 50
    patience: int = 10
    minibatch_size: int = 5
    seed: int = 42
    output_dir: str = "outputs/gaia"
