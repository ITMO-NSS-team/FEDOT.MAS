from pydantic_settings import BaseSettings, SettingsConfigDict


class HotpotQASettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HOTPOT_")

    solver_model: str = "openai/gpt-4.1-mini"
    max_iterations: int = 10_000
    max_evaluations: int | None = None
    patience: int = 10_000
    minibatch_size: int = 3
    use_merge: bool = False
    seed: int = 42
    output_dir: str = "outputs/hotpot_qa"
    train_limit: int | None = None
    val_limit: int | None = None
    test_limit: int | None = None
    test_repeats: int = 1
    concurrency: int = 8
    eval_concurrency: int = 8
    max_output_tokens: int | None = None
    skip_baseline_test: bool = False
    baseline_accuracy: float | None = None
    eval_best_on_train: bool = False
    checkpoint_path: str | None = None
