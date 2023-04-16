from typing import List


class ScoringFuncionParameters:
    def __init__(
        self, name: str,
        parameters: List[dict],
        parallel=False,
        n_cpu=4,
        custom_scorers: List[str] = [],
        custom_scorer_weights: List[float] = []
    ):
        self.name = name
        self.parameters = parameters
        self.parallel = parallel
        self.n_cpu = n_cpu
        self.custom_scorers = custom_scorers
        self.custom_scorer_weights = custom_scorer_weights
