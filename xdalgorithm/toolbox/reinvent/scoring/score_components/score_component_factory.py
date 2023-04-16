from typing import List

from xdalgorithm.toolbox.reinvent.scoring.component_parameters import ComponentParameters
from xdalgorithm.toolbox.reinvent.scoring.score_components import (
    TanimotoSimilarity,
    JaccardDistance,
    CustomAlerts,
    QedScore,
    MatchingSubstructure,
    PredictivePropertyComponent,
    SelectivityComponent,
    # SASComponent,
    MolWeight,
    PSA,
    RotatableBonds,
    HBD_Lipinski,
    NumRings,
    ExcludedVolume,
    Pharmacophore_Align,
    HBA_Lipinski,
    Shape,
    PharmacophoreShapeCombination,
    PharmacophoreConstraint,
    CustomMust
)
from xdalgorithm.toolbox.reinvent.scoring.score_components.base_score_component import BaseScoreComponent
from xdalgorithm.toolbox.reinvent.utils.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum


class ScoreComponentFactory:
    def __init__(self, parameters: List[ComponentParameters]):
        self._parameters = parameters
        self._current_components = self._deafult_scoring_component_registry()

    def _deafult_scoring_component_registry(self) -> dict:
        enum = ScoringFunctionComponentNameEnum()
        component_map = {
            enum.MATCHING_SUBSTRUCTURE: MatchingSubstructure,
            enum.PREDICTIVE_PROPERTY: PredictivePropertyComponent,
            enum.TANIMOTO_SIMILARITY: TanimotoSimilarity,
            enum.JACCARD_DISTANCE: JaccardDistance,
            enum.CUSTOM_ALERTS: CustomAlerts,
            enum.QED_SCORE: QedScore,
            enum.MOLECULAR_WEIGHT: MolWeight,
            enum.TPSA: PSA,
            enum.NUM_ROTATABLE_BONDS: RotatableBonds,
            enum.NUM_HBD_LIPINSKI: HBD_Lipinski,
            enum.NUM_RINGS: NumRings,
            enum.SELECTIVITY: SelectivityComponent,
            # enum.SA_SCORE: SASComponent,
            enum.EXCLUDED_VOLUME: ExcludedVolume,
            enum.PHARMACOPHORE_ALIGN: Pharmacophore_Align,
            enum.NUM_HBA_LIPINSKI: HBA_Lipinski,
            enum.SHAPE: Shape,
            enum.PHARMACOPHORE_SHAPE_COMBINATION: PharmacophoreShapeCombination,
            enum.PHARMACOPHORE_CONSTRAINT: PharmacophoreConstraint,
            enum.CUSTOM_MUST: CustomMust
        }
        return component_map

    def create_score_components(self) -> List[BaseScoreComponent]:
        score_components_list = []
        for p in self._parameters:
            if p.component_type in self._current_components.keys():
                score_components_list.append(self._current_components.get(p.component_type)(p))
            else:
                raise ValueError("Component type {0} is not supported!".format(p.component_type))
        return score_components_list
