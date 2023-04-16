from typing import List
from xdalgorithm.toolbox.reinvent.scoring.component_parameters import ComponentParameters
from xdalgorithm.toolbox.reinvent.scoring.score_components.base_score_component import BaseScoreComponent
from xdalgorithm.toolbox.reinvent.scoring.score_summary import ComponentSummary
from xdalgorithm.toolbox.scoring_component_library.custom_must import CustomMust as CustomMustScorer

class CustomMust(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        # load operator
        operator = self.parameters.specific_parameters.get("operator")
        smarts_list = self.parameters.smiles
        self.scorer = CustomMustScorer(
            operator=operator,
            smarts_list=smarts_list
        )

    def calculate_score(self, molecules: List) -> ComponentSummary:
        score = [self.scorer.calculate_score(molecule) for molecule in molecules]
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters)
        return score_summary

    def get_component_type(self):
        return "custom_must"
