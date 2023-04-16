from xdalgorithm.toolbox.scoring_component_library.synthetic_accessibility.sas_component import \
    SASComponent as SASScorer
from typing import List
from xdalgorithm.toolbox.reinvent.scoring.component_parameters import ComponentParameters
from xdalgorithm.toolbox.reinvent.scoring.score_components.base_score_component import BaseScoreComponent
from xdalgorithm.toolbox.reinvent.scoring.score_summary import ComponentSummary


class SASComponent(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        model_path = parameters.model_path
        self.scorer = SASScorer(model_path=model_path)

    def calculate_score(self, molecules: List) -> ComponentSummary:
        scores = [self.scorer.calculate_score(molecule) for molecule in molecules]
        score_summary = ComponentSummary(total_score=scores, parameters=self.parameters)
        return score_summary
