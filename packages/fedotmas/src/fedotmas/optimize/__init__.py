from __future__ import annotations

from fedotmas.common.logging import get_logger
from fedotmas.maw.maw import MAW
from fedotmas.maw.models import MAWConfig
from fedotmas.optimize._engine import run_optimization
from fedotmas.optimize._proposer import Proposer
from fedotmas.optimize._result import OptimizationResult
from fedotmas.optimize._scoring import LLMJudge, Scorer, ScoringResult
from fedotmas.optimize._state import Candidate
from fedotmas.optimize._stopping import (
    CompositeStopper,
    MaxEvaluations,
    MaxIterations,
    NoImprovement,
    ScoreThreshold,
    Stopper,
)
from fedotmas.optimize._strategies import (
    ShuffledBatchSampler,
    make_candidate_selector,
    make_component_selector,
)

_log = get_logger("fedotmas.optimize")

__all__ = [
    "Optimizer",
    "OptimizationResult",
    "Scorer",
    "ScoringResult",
    "LLMJudge",
    "Candidate",
]


class Optimizer:
    def __init__(
        self,
        maw: MAW,
        *,
        scorer: Scorer | None = None,
        criteria: str | None = None,
        candidate_selection: str = "pareto",
        use_merge: bool = True,
        max_merge_attempts: int = 5,
        minibatch_size: int = 3,
    ) -> None:
        self._maw = maw
        self._use_merge = use_merge
        self._max_merge_attempts = max_merge_attempts
        self._minibatch_size = minibatch_size

        if scorer is not None:
            self._scorer: Scorer = scorer
        else:
            self._scorer = LLMJudge(criteria=criteria)

        self._candidate_selector = make_candidate_selector(candidate_selection)
        self._batch_sampler = ShuffledBatchSampler()
        self._last_result: OptimizationResult | None = None

    async def optimize(
        self,
        trainset: list[str],
        *,
        seed_config: MAWConfig | None = None,
        valset: list[str] | None = None,
        max_iterations: int = 20,
        max_evaluations: int | None = None,
        patience: int = 5,
        score_threshold: float | None = None,
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
            seed_config = await self._maw.generate_config(trainset[0])

        component_selector = make_component_selector(len(seed_config.agents))

        stoppers: list[Stopper] = [MaxIterations(max_iterations)]
        if max_evaluations is not None:
            stoppers.append(MaxEvaluations(max_evaluations))
        stoppers.append(NoImprovement(patience))
        if score_threshold is not None:
            stoppers.append(ScoreThreshold(score_threshold))
        stopper = CompositeStopper(stoppers)

        proposer = Proposer()

        result = await run_optimization(
            maw=self._maw,
            seed_config=seed_config,
            trainset=trainset,
            valset=valset,
            scorer=self._scorer,
            proposer=proposer,
            candidate_selector=self._candidate_selector,
            batch_sampler=self._batch_sampler,
            component_selector=component_selector,
            stopper=stopper,
            use_merge=self._use_merge,
            max_merge_attempts=self._max_merge_attempts,
            minibatch_size=self._minibatch_size,
        )

        prompt, completion = proposer.token_usage
        result.total_prompt_tokens += prompt
        result.total_completion_tokens += completion

        scorer_usage = getattr(self._scorer, "token_usage", None)
        if isinstance(scorer_usage, tuple) and len(scorer_usage) == 2:
            result.total_prompt_tokens += scorer_usage[0]
            result.total_completion_tokens += scorer_usage[1]

        self._last_result = result
        return result

    @property
    def last_result(self) -> OptimizationResult | None:
        return self._last_result
