from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class OptimizationConfig:
    """Configuration for the evolutionary optimization loop."""

    # Stopping criteria
    max_iterations: int = 20
    """Maximum number of optimization iterations."""

    patience: int = 5
    """Stop after this many iterations without score improvement."""

    score_threshold: float | None = None
    """Stop when best candidate reaches this score. None = disabled."""

    max_evaluations: int | None = None
    """Stop after this many total evaluation runs. None = disabled."""

    # LLM temperatures
    temperature_reflect: float = 0.7
    """LLM temperature for reflection (mutation) calls. Range: [0, 2]."""

    temperature_merge: float = 0.5
    """LLM temperature for merge calls. Range: [0, 2]."""

    temperature_judge: float = 0.1
    """LLM temperature for scoring/judgment. Low = consistent. Range: [0, 2].
    Ignored if custom Scorer is provided."""

    # LLM safety
    llm_timeout: float = 120.0
    """Timeout in seconds for each LLM call (reflect, merge, judge). 0 = no timeout."""

    # Evolutionary operators
    candidate_selection: Literal["pareto", "best", "epsilon_greedy"] = "pareto"
    """Strategy for selecting parent candidate each iteration.
    - "pareto": random from Pareto front (balanced exploration)
    - "best": always pick highest mean score (greedy)
    - "epsilon_greedy": 10% random, 90% best
    """

    use_merge: bool = True
    """Enable merging of Pareto candidates (crossover operator)."""

    max_merge_attempts: int = 5
    """Maximum merge operations per optimization run."""

    minibatch_size: int = 3
    """Number of tasks sampled per iteration for evaluation."""

    batch_strategy: Literal["epoch_shuffled", "random"] = "epoch_shuffled"
    """Batch sampling strategy.
    - "epoch_shuffled": sequential minibatches from shuffled trainset,
      guarantees full coverage per epoch (GEPA default)
    - "random": random sample each iteration, no coverage guarantee
    """

    # Truncation limits
    max_merge_context_tasks: int = 5
    """Max tasks included in merge LLM context."""

    max_state_chars: int = 2000
    """Max characters of pipeline state passed to scorer."""

    max_output_chars: int = 3000
    """Max characters of agent output in reflection examples."""

    # Mutation types (toggle which aspects of the graph to optimize)
    mutate_instructions: bool = True
    """Optimize agent instructions via LLM reflection."""

    mutate_tools: bool = False
    """Optimize agent tool assignments. (Not yet implemented)"""

    mutate_models: bool = False
    """Optimize model selection per agent. (Not yet implemented)"""

    mutate_structure: bool = False
    """Optimize pipeline structure (agent ordering, parallel/sequential). (Not yet implemented)"""

    # Error recovery
    max_consecutive_failures: int = 3
    """Threshold before emergency shuffle of agents."""

    improvement_epsilon: float = 1e-6
    """Minimum score delta to count as improvement for NoImprovement stopper.
    NOT related to epsilon-greedy exploration."""

    # Infrastructure
    checkpoint_path: str | None = None
    """Path to save/restore optimization state. None = no checkpointing."""

    graceful_shutdown: bool = False
    """Install SIGINT/SIGTERM handlers for graceful exit after current iteration."""

    seed: int | None = None
    """Random seed for reproducibility."""

    _rng: random.Random = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    @property
    def rng(self) -> random.Random:
        return self._rng
