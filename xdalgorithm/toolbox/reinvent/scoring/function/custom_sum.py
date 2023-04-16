from typing import List

import numpy as np

from xdalgorithm.toolbox.reinvent.scoring.component_parameters import ComponentParameters
from xdalgorithm.toolbox.reinvent.scoring.function.base_scoring_function import BaseScoringFunction
from xdalgorithm.toolbox.reinvent.scoring.score_summary import ComponentSummary


class CustomSum(BaseScoringFunction):

    def __init__(self, 
                 parameters: List[ComponentParameters],
                 parallel=False,
                 n_cpu=4,
                 custom_scorers=[],
                 custom_scorer_weights=[]):
        super().__init__(parameters,
                         parallel,
                         n_cpu=n_cpu,
                         custom_scorers=custom_scorers,
                         custom_scorer_weights=custom_scorer_weights)

    def _compute_non_penalty_components(self, summaries: List[ComponentSummary], smiles: List[str]):
        total_sum = np.zeros(len(smiles))
        all_weights = 0.

        for summary in summaries:
            if not self._component_is_penalty(summary):
                total_sum = total_sum + summary.total_score * summary.parameters.weight
                all_weights += summary.parameters.weight

        if all_weights == 0:
            """There are no non-penalty components and return array of ones. 
            This is needed so that it can work in cases where only penalty components are used"""
            return np.ones(len(smiles))
        total_sum = (total_sum + np.abs(total_sum)) / 2.0
        return total_sum / all_weights
