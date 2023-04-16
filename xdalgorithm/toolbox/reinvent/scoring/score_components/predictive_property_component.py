from typing import List
from xdalgorithm.toolbox.reinvent.scoring.component_parameters import ComponentParameters
from xdalgorithm.toolbox.reinvent.scoring.score_components.base_score_component import BaseScoreComponent
from xdalgorithm.toolbox.reinvent.scoring.score_summary import ComponentSummary
from xdalgorithm.toolbox.scoring_component_library.predictive_property import PredictiveProperty

class PredictivePropertyComponent(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        model_type = self.parameters.specific_parameters['scikit']
        model_path = self.parameters.model_path
        parameters_config = self.parameters.specific_parameters
        self.scorer = PredictiveProperty(model_type=model_type,
                                         model_path=model_path,
                                         parameters_config=parameters_config)

    def calculate_score(self, molecules: List) -> ComponentSummary:
        score = [self.scorer.calculate_score(molecule) for molecule in molecules]
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters)
        return score_summary

    def get_component_type(self):
        return "predictive_property"
