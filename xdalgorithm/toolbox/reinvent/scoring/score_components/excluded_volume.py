from rdkit import Chem
from typing import List
from xdalgorithm.toolbox.reinvent.scoring.component_parameters import ComponentParameters
from xdalgorithm.toolbox.reinvent.scoring.score_components.base_score_component import BaseScoreComponent
from xdalgorithm.toolbox.reinvent.scoring.score_summary import ComponentSummary
from xdalgorithm.toolbox.scoring_component_library.excluded_volume import \
    ExcludedVolume as ExcludedVolumeScorer


class ExcludedVolume(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super(ExcludedVolume, self).__init__(parameters)
        self.template_mol = [m for m in Chem.SDMolSupplier(self.parameters.specific_parameters["template_mol_file"])][0]
        self.ligand_name = self.parameters.specific_parameters["lig_name"]
        self.pdb_name = self.parameters.specific_parameters["pdb_name"]
        self.atom_width = self.parameters.specific_parameters["atom_width"]
        self.penalty_weight = self.parameters.specific_parameters["penalty_weight"]

        self.excluded_volume = ExcludedVolumeScorer(
            ligand_name=self.ligand_name,
            pdb_name=self.pdb_name,
            atom_width=self.atom_width,
            penalty_weight=self.penalty_weight,
            template_mol=self.template_mol
        )

    def calculate_score(self, molecules: List) -> ComponentSummary:
        scores = [self.excluded_volume.calculate_score(molecule) for molecule in molecules]
        score_summary = ComponentSummary(total_score=scores, parameters=self.parameters)
        return score_summary

    def get_component_type(self):
        return "excluded_volume"
