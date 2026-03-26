from __future__ import annotations

from fedotmas.common.logging import get_logger
from fedotmas.maw.maw import MAW
from fedotmas.maw.models import MAWConfig
from fedotmas.optimize._callbacks import OptimizationCallback
from fedotmas.optimize._config import OptimizationConfig
from fedotmas.optimize._engine import run_optimization
from fedotmas.optimize._mutators._instruction import InstructionMutator
from fedotmas.optimize._mutators._protocol import Mutator
from fedotmas.optimize._result import OptimizationResult
from fedotmas.optimize._scoring import LLMJudge, Scorer
from fedotmas.optimize._state import Task
from fedotmas.optimize._stopping import (
    CompositeStopper,
    MaxEvaluations,
    MaxIterations,
    NoImprovement,
    ScoreThreshold,
    Stopper,
)
from fedotmas.optimize._strategies import (
    make_batch_sampler,
    make_candidate_selector,
    make_component_selector,
)

_log = get_logger("fedotmas.optimize._optimizer")


class Optimizer:
    def __init__(
        self,
        maw: MAW,
        *,
        scorer: Scorer | None = None,
        criteria: str | None = None,
        config: OptimizationConfig | None = None,
        callbacks: list[OptimizationCallback] | None = None,
        mutator: Mutator | None = None,
    ) -> None:
        self._config: OptimizationConfig = config or OptimizationConfig()
        self._maw = maw
        self._callbacks = callbacks or []
        self._mutator = mutator

        if scorer is not None:
            self._scorer: Scorer = scorer
        else:
            self._scorer = LLMJudge(
                criteria=criteria,
                max_state_chars=self._config.max_state_chars,
                temperature=self._config.temperature_judge,
            )

        rng = self._config.rng
        self._candidate_selector = make_candidate_selector(
            self._config.candidate_selection, rng=rng
        )
        self._batch_sampler = make_batch_sampler(self._config.batch_strategy, rng=rng)
        self._last_result: OptimizationResult | None = None

    async def optimize(
        self,
        trainset: list[Task],
        *,
        seed_config: MAWConfig | None = None,
        valset: list[Task] | None = None,
    ) -> OptimizationResult:
        if not trainset:
            raise ValueError("trainset must not be empty")

        if valset is None:
            _log.warning(
                "No separate valset provided; using trainset for validation. "
                "Risk of overfitting."
            )
            valset = trainset

        if seed_config is None:
            _log.info("Generating seed config from first task")
            seed_config = await self._maw.generate_config(trainset[0].input)

        component_selector = make_component_selector(len(seed_config.agents))

        cfg = self._config
        stoppers: list[Stopper] = [MaxIterations(cfg.max_iterations)]
        if cfg.max_evaluations is not None:
            stoppers.append(MaxEvaluations(cfg.max_evaluations))
        stoppers.append(NoImprovement(cfg.patience, epsilon=cfg.improvement_epsilon))
        if cfg.score_threshold is not None:
            stoppers.append(ScoreThreshold(cfg.score_threshold))
        stopper = CompositeStopper(stoppers)

        mutator = self._mutator or InstructionMutator(self._config)

        result = await run_optimization(
            maw=self._maw,
            seed_config=seed_config,
            trainset=trainset,
            valset=valset,
            scorer=self._scorer,
            mutator=mutator,
            candidate_selector=self._candidate_selector,
            batch_sampler=self._batch_sampler,
            component_selector=component_selector,
            stopper=stopper,
            config=self._config,
            callbacks=self._callbacks,
        )

        scorer_usage = getattr(self._scorer, "token_usage", None)
        if isinstance(scorer_usage, tuple) and len(scorer_usage) == 2:
            result.total_prompt_tokens += scorer_usage[0]
            result.total_completion_tokens += scorer_usage[1]

        self._last_result = result
        return result

    @property
    def last_result(self) -> OptimizationResult | None:
        return self._last_result
