from typing import List
from xdalgorithm.toolbox.reinvent.scoring.component_parameters import ComponentParameters
from xdalgorithm.toolbox.reinvent.scoring.score_components.base_score_component import BaseScoreComponent
from xdalgorithm.toolbox.reinvent.scoring.score_summary import ComponentSummary
from xdalgorithm.toolbox.scoring_component_library.substructure_matching \
    import SubstructureMatching as SubstructureMatchingScorer

class MatchingSubstructure(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self.target_smarts = self.parameters.smiles  # these are actually smarts
        self.scorer = SubstructureMatchingScorer(self.target_smarts)

    def calculate_score(self, molecules: List) -> ComponentSummary:
        score = [self.scorer.calculate_score(molecule) for molecule in molecules]
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters)
        return score_summary

    def get_component_type(self):
        return "matching_substructure"
