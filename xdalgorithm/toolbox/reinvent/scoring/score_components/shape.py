import typing as t
from rdkit import Chem
from xdalgorithm.toolbox.reinvent.scoring.component_parameters import ComponentParameters
from xdalgorithm.toolbox.reinvent.scoring.score_components.base_score_component import BaseScoreComponent
from xdalgorithm.toolbox.reinvent.scoring.score_summary import ComponentSummary
from xdalgorithm.toolbox.scoring_component_library.shape import Shape as ShapeScorer

__all__ = [
    'Shape',
]


class Shape(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self.ref_sdf = self.parameters.specific_parameters["ref_sdf"]
        self.ref_mol = Chem.SDMolSupplier(self.ref_sdf, removeHs=False)[0]
        self.max_confshapes = self.parameters.specific_parameters["max_confshapes"]
        self.use_chemaxon = self.parameters.specific_parameters.get('use_chemaxon', False)
        self.protonation = self.parameters.specific_parameters.get('protonation', True)
        self.shape_scorer = ShapeScorer(template_mol=self.ref_mol,
                                        max_confshapes=self.max_confshapes,
                                        num_threads=12,
                                        use_chemaxon=self.use_chemaxon,
                                        protonation=self.protonation)

    def calculate_score(self, molecules: t.List) -> ComponentSummary:
        scores = [self.shape_scorer.calculate_score(molecule) for molecule in molecules]
        score_summary = ComponentSummary(total_score=scores,
                                         parameters=self.parameters)
        return score_summary

    def get_component_type(self):
        return "shape"
