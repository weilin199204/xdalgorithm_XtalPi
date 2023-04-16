from rdkit import Chem
from typing import List
from xdalgorithm.toolbox.reinvent.scoring.score_summary import ComponentSummary
from xdalgorithm.toolbox.reinvent.scoring.component_parameters import ComponentParameters
from xdalgorithm.toolbox.reinvent.scoring.score_components.base_score_component import BaseScoreComponent
from xdalgorithm.toolbox.scoring_component_library.pharmacophore_constraint import P4coreConstraint

class PharmacophoreConstraint(BaseScoreComponent):
	def __init__(self, parameters: ComponentParameters):
		super(PharmacophoreConstraint, self).__init__(parameters)
		self.refMol = [m for m in Chem.SDMolSupplier(self.parameters.specific_parameters["ref_mol"])][0]

		self.weightsPolicy = self.parameters.specific_parameters["weights_policy"]		
		self.validP4cores = self.parameters.specific_parameters["valid_p4cores"]
		self.maxNumP4cores = self.parameters.specific_parameters["max_num_p4cores"]
		self.seleP4cores = self.parameters.specific_parameters["sele_p4cores"]
		if "sele_p4cores_weights" in self.parameters.specific_parameters:
			self.seleP4coresWeights = self.parameters.specific_parameters["sele_p4cores_weights"]
		else:
			self.seleP4coresWeights = [1.] * len(self.seleP4cores)

		self.atomicRadius = self.parameters.specific_parameters["atomic_radius"]
		self.maxToTryEmbedP4core = self.parameters.specific_parameters["max_to_try_embed_p4core"]
		self.numDesiredEmbedP4core = self.parameters.specific_parameters["num_desired_embed_p4core"]
		self.dUpper = self.parameters.specific_parameters["d_upper"]
		self.dLower = self.parameters.specific_parameters["d_lower"]
		self.tmpdir = self.parameters.specific_parameters["tmpdir"] if "tmpdir" in self.parameters.specific_parameters else None

		self.scorer = P4coreConstraint(
			weightsPolicy = self.weightsPolicy,
			refMol = self.refMol,
			validP4cores = self.validP4cores,
			maxNumP4cores = self.maxNumP4cores,
			seleP4cores = self.seleP4cores,
			seleP4coresWeights = self.seleP4coresWeights,
			atomicRadius = self.atomicRadius,
			maxToTryEmbedP4core = self.maxToTryEmbedP4core,
			numDesiredEmbedP4core = self.numDesiredEmbedP4core,
			dUpper = self.dUpper,
			dLower = self.dLower,
			tmpdir = self.tmpdir
		)

	def calculate_score(self, molecules: List) -> ComponentSummary:
		scores = [self.scorer.calculate_score(molecule) for molecule in molecules]
		score_summary = ComponentSummary(total_score=scores, parameters=self.parameters)
		return score_summary
	
	def get_component_type(self):
		return "pharmacophore_constraint"