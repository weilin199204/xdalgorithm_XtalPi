import os
from rdkit import Chem
from rdkit.Chem import AllChem
from xdalgorithm.toolbox.reinvent.scoring.score_components.base_score_component import BaseScoreComponent
from xdalgorithm.toolbox.reinvent.scoring.component_parameters import ComponentParameters
from xdalgorithm.toolbox.reinvent.scoring.score_summary import ComponentSummary
from xdalgorithm.toolbox.scoring_component_library.pharmacophore_align \
    import PharmacophoreAlign as PharmacophoreAlignScorer


class Pharmacophore_Align(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super(Pharmacophore_Align, self).__init__(parameters)
        self.template_mol = [m for m in Chem.SDMolSupplier(self.parameters.specific_parameters["template_mol_file"])][0]

        import xdalgorithm
        predefinedBaseFeatFilepath = os.path.join(
            os.path.dirname(xdalgorithm.__file__), 
            "toolbox/interaction_fingerprints/data/BaseFeatures.fdef")
        self.fdef = AllChem.BuildFeatureFactory(predefinedBaseFeatFilepath)

        self.d_upper = self.parameters.specific_parameters["d_upper"]
        self.d_lower = self.parameters.specific_parameters["d_lower"]
        self.keep = self.parameters.specific_parameters["keep"]

        self.pList_max_allowed = self.parameters.specific_parameters["pList_max_allowed"]
        self.max_to_try = self.parameters.specific_parameters["max_to_try"]
        self.num_desired = self.parameters.specific_parameters["num_desired"]
        self.reward_weights = self.parameters.specific_parameters["stepwise_reward_weights"]
        self.pharmacophore_idxs = self.parameters.specific_parameters["pharmacophore_idxs"]
        self.atom_width = self.parameters.specific_parameters["atom_width"]
        self.use_chemaxon = self.parameters.specific_parameters.get('use_chemaxon', False)
        self.protonation = self.parameters.specific_parameters.get('protonation', True)

        if "pharmacophore_weights" in self.parameters.specific_parameters:
            self.pharmacophore_weights = self.parameters.specific_parameters["pharmacophore_weights"]
        else:
            self.pharmacophore_weights = [1.] * len(self.pharmacophore_idxs)

        if "penalty_weight" in self.parameters.specific_parameters:
            self.lig_name = self.parameters.specific_parameters["lig_name"]
            self.pdb_name = self.parameters.specific_parameters["pdb_name"]
            self.penalty_weight = self.parameters.specific_parameters["penalty_weight"]
            self.scorer = PharmacophoreAlignScorer(reward_weights=self.reward_weights,
                                                   d_upper=self.d_upper,
                                                   d_lower=self.d_lower,
                                                   keep=self.keep,
                                                   pharmacophore_max_allowed=self.pList_max_allowed,
                                                   max_to_try=self.max_to_try,
                                                   num_desired=self.num_desired,
                                                   pharmacophore_idxs=self.pharmacophore_idxs,
                                                   template_mol=self.template_mol,
                                                   atom_width=self.atom_width,
                                                   pharmacophore_weights=self.pharmacophore_weights,
                                                   excluded_volume=True,
                                                   lig_name=self.lig_name,
                                                   pdb_name=self.pdb_name,
                                                   penalty_weight=self.penalty_weight,
                                                   use_chemaxon=self.use_chemaxon,
                                                   protonation=self.protonation)
        else:
            self.scorer = PharmacophoreAlignScorer(reward_weights=self.reward_weights,
                                                   d_upper=self.d_upper,
                                                   d_lower=self.d_lower,
                                                   keep=self.keep,
                                                   pharmacophore_max_allowed=self.pList_max_allowed,
                                                   max_to_try=self.max_to_try,
                                                   num_desired=self.num_desired,
                                                   pharmacophore_idxs=self.pharmacophore_idxs,
                                                   template_mol=self.template_mol,
                                                   atom_width=self.atom_width,
                                                   pharmacophore_weights=self.pharmacophore_weights,
                                                   excluded_volume=False)

    def calculate_score(self, molecules) -> ComponentSummary:
        scores = [self.scorer.calculate_score(molecule) for molecule in molecules]
        score_summary = ComponentSummary(total_score=scores, parameters=self.parameters)
        return score_summary

    def get_component_type(self):
        return "pharmacophore_align"
