from pydantic_settings import BaseSettings, SettingsConfigDict


class AimeMathSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AIME_")

    solver_model: str = "openai/gpt-4.1-mini"
    max_iterations: int = 50
    patience: int = 10
    minibatch_size: int = 5
    seed: int = 42
    output_dir: str = "outputs/aime_math"
