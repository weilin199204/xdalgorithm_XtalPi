from typing import List
from xdalgorithm.toolbox.reinvent.scoring.component_parameters import ComponentParameters
from xdalgorithm.toolbox.reinvent.scoring.score_components.base_score_component import BaseScoreComponent
from xdalgorithm.toolbox.reinvent.scoring.score_summary import ComponentSummary
from xdalgorithm.toolbox.scoring_component_library.tanimoto_similarity \
    import TanimotoSimilarity as TanimotoSimilarityScorer


class TanimotoSimilarity(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self.scorer = TanimotoSimilarityScorer(smiles_list=self.parameters.smiles,
                                               radius=3,
                                               use_counts=True,
                                               use_features=True)

    def calculate_score(self, molecules: List) -> ComponentSummary:
        score = [self.scorer.calculate_score(molecule) for molecule in molecules]
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters)
        return score_summary

    def get_component_type(self):
        return "tanimoto_similarity"
