from abc import ABC, abstractmethod
from typing import List, Tuple
from functools import partial
from pathos.multiprocessing import ProcessPool
import numpy as np
import typing as t
from rdkit import Chem

from xdalgorithm.toolbox.reinvent.scoring.component_parameters import ComponentParameters
from xdalgorithm.toolbox.reinvent.scoring.score_components.score_component_factory import ScoreComponentFactory
from xdalgorithm.toolbox.reinvent.scoring.score_summary import ComponentSummary, FinalSummary
from xdalgorithm.toolbox.reinvent.utils.enums.component_specific_parameters_enum import ComponentSpecificParametersEnum
from xdalgorithm.toolbox.reinvent.utils.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum

from xdalgorithm.toolbox.reinvent.scoring.score_components import TanimotoSimilarity, \
    JaccardDistance, CustomAlerts, QedScore, MatchingSubstructure, \
    PredictivePropertyComponent, SelectivityComponent, \
    SASComponent, MolWeight, PSA, RotatableBonds, HBD_Lipinski, NumRings, \
    ExcludedVolume, Pharmacophore_Align, \
    HBA_Lipinski, Shape, PharmacophoreShapeCombination, PharmacophoreConstraint, \
    CustomMust
from xdalgorithm.toolbox.reinvent.scoring.function.custom_score import get_func_from_dir


def scoring_component_registry(component_type, component_parameters):
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
        enum.SA_SCORE: SASComponent,
        enum.EXCLUDED_VOLUME: ExcludedVolume,
        enum.PHARMACOPHORE_ALIGN: Pharmacophore_Align,
        enum.NUM_HBA_LIPINSKI: HBA_Lipinski,
        enum.SHAPE: Shape,
        enum.PHARMACOPHORE_SHAPE_COMBINATION: PharmacophoreShapeCombination,
        enum.PHARMACOPHORE_CONSTRAINT: PharmacophoreConstraint,
        enum.CUSTOM_MUST: CustomMust
    }
    return component_map.get(component_type)(component_parameters)


def _update_total_score(summary: ComponentSummary, query_length: int, valid_indices: List[int]) -> ComponentSummary:
    total_score = np.full(query_length, 0, dtype=np.float32)
    assert len(valid_indices) == len(summary.total_score)
    for idx, value in zip(valid_indices, summary.total_score):
        total_score[idx] = value
    summary.total_score = total_score
    return summary


# one component, one molecule, return a score
def create_and_calculate_scores(component_config, molecule):
    component_type, component_parameters = component_config
    component = scoring_component_registry(component_type, component_parameters)
    score_component_summary = component.calculate_score([molecule])  # return ComponentSummary
    return score_component_summary.total_score[0]


def parrallel_by_molecule(molecule_configs, components_configs):
    molecule_idx, molecule = molecule_configs
    component_scores = []
    for component_config in components_configs:
        try:
            component_score = create_and_calculate_scores(component_config, molecule)
            component_scores.append(component_score)
        except:
            print(f'ERROR: {Chem.MolToSmiles(molecule)}\n')
            component_scores.append(0)
    return molecule_idx, component_scores

################################################

def parallel_run(component_smiles_pair):
    component_type, component_parameters = component_smiles_pair[0]
    component = scoring_component_registry(component_type, component_parameters)
    molecules = component_smiles_pair[1]
    valid_indices = component_smiles_pair[2]
    smiles = component_smiles_pair[3]
    scores = component.calculate_score(molecules)
    scores = _update_total_score(scores, len(smiles), valid_indices)
    return scores


class BaseScoringFunction(ABC):
    def __init__(
        self,
        parameters: List[ComponentParameters],
        parallel=False,
        n_cpu=4,
        custom_scorers: List[str] = [],
        custom_scorer_weights: List[float] = []
    ):
        self.component_enum = ScoringFunctionComponentNameEnum()
        self.component_specific_parameters = ComponentSpecificParametersEnum()
        self.n_cpu = n_cpu
        print("cpu_num={0}".format(self.n_cpu))
        factory = ScoreComponentFactory(parameters)
        self.scoring_components = factory.create_score_components()
        self.parallel = parallel
        if parallel:
            self.get_final_score = self._parallel_final_score

        custom_score_dirs = custom_scorers
        if isinstance(custom_score_dirs, str):
            custom_score_dirs = [custom_score_dirs]
        if not isinstance(custom_scorer_weights, t.Iterable):
            custom_scorer_weights = [custom_scorer_weights] * len(custom_score_dirs)
        # print(custom_scorers)
        custom_scores = [
            get_func_from_dir(score_dir)
            for score_dir in custom_score_dirs
        ]
        self.custom_score_funcs = [
            score[0]
            for score in custom_scores
        ]
        # print(self.custom_score_funcs)
        self.custom_score_modes = [
            score[1].lower()
            for score in custom_scores
        ]
        self.custom_scorer_weights = custom_scorer_weights
 
    def get_valid_smiles(
        self,
        smiles_list: List[str]
    ):
        valid_mols, valid_indices = self._smiles_to_mols(smiles_list) 
        valid_smiles = list(map(smiles_list.__getitem__, valid_indices)) 
        return valid_mols, valid_smiles, valid_indices

    def get_custom_scores(
        self,
        smiless: List[str]
    ) -> List[float]:
        _, valid_indices = self._smiles_to_mols(smiless)
        # valid_smiles = list(map(smiless.__getitem__, valid_indices))
        score_sum: np.ndarray = np.zeros(len(smiless))
        for scorer, score_mode, score_weight in zip(
            self.custom_score_funcs,
            self.custom_score_modes,
            self.custom_scorer_weights
        ):
            if score_mode == 'batch':
                scores = np.array(scorer(smiless))
            else:
                if not self.parallel:
                    scores = np.array([
                        scorer(smiles)
                        for smiles in smiless
                    ])
                elif self.parallel:
                    scores = np.zeros(len(smiless))
            score_sum += scores * score_weight
        # score_sum_list: List = score_sum.tolist()
        return score_sum 

    def get_final_score(self, smiles: List[str]) -> FinalSummary:
        molecules, valid_indices = self._smiles_to_mols(smiles)
        query_size = len(smiles)
        summaries = [_update_total_score(sc.calculate_score(molecules), query_size, valid_indices) for sc
                     in self.scoring_components]
        return self._score_summary(summaries, smiles, valid_indices)

    def _score_summary(self, summaries: List[ComponentSummary], smiles: List[str],
                       valid_indices: List[int]) -> FinalSummary:

        penalty = self._compute_penalty_components(summaries, smiles)  # a list consist of 1 and 0
        non_penalty = self._compute_non_penalty_components(summaries, smiles)
        product = penalty * non_penalty
        product += self.get_custom_scores(smiless=smiles)
        final_summary = self._create_final_summary(product, summaries, smiles, summaries, valid_indices)

        return final_summary

    def _create_final_summary(self, final_score, summaries: List[ComponentSummary], smiles: List[str],
                              log_summary: List[ComponentSummary], valid_indices: List[int]) -> FinalSummary:

        return FinalSummary(total_score=np.array(final_score, dtype=np.float32),
                            scored_smiles=smiles,
                            valid_idxs=valid_indices,
                            scaffold_log_summary=summaries,
                            log_summary=log_summary)

    def _compute_penalty_components(self, summaries: List[ComponentSummary], smiles: List[str]):
        # the smiles matched any smarts gets zero,else 1.
        penalty = np.ones(len(smiles))

        for summary in summaries:
            if self._component_is_penalty(summary):
                penalty = penalty * summary.total_score

        return penalty

    @abstractmethod
    def _compute_non_penalty_components(self, summaries: List[ComponentSummary], smiles: List[str]):
        raise NotImplementedError("_score_summary method is not implemented")

    def _component_is_penalty(self, summary: ComponentSummary) -> bool:
        return (summary.parameters.component_type == self.component_enum.MATCHING_SUBSTRUCTURE) or (
                summary.parameters.component_type == self.component_enum.CUSTOM_ALERTS) or (
                summary.parameters.component_type == self.component_enum.CUSTOM_MUST)

    def _parallel_final_score(self, smiles: List[str]) -> FinalSummary:
        molecules, valid_indices = self._smiles_to_mols(smiles)
        components_configs = [(component.get_component_type(), component.get_parameters())
                              for component in self.scoring_components]
        print("start multiprocessing...")
        pool = ProcessPool(nodes=self.n_cpu)
        parallel_fun = partial(parrallel_by_molecule, components_configs=components_configs)
        molecules_configs = [(i, mol) for i, mol in enumerate(molecules)]
        mapped_pool = pool.map(parallel_fun, molecules_configs)  # molecule_num * scoring_components_num
        # sort result in order
        order_mapped_pool = [None for _ in range(len(mapped_pool))]
        for i, map_score in mapped_pool:
            order_mapped_pool[i] = map_score

        pool.clear()
        print("complete multiprocessing")
        scores_matrix = np.array(order_mapped_pool).transpose()  # scoring_components_num * molecule_num
        score_summary_list = []
        for i in range(len(components_configs)):
            scores_summary = ComponentSummary(total_score=scores_matrix[i,:], parameters=components_configs[i][1])
            full_summary = _update_total_score(scores_summary,len(smiles), valid_indices)
            score_summary_list.append(full_summary)
        return self._score_summary(score_summary_list, smiles, valid_indices)

    def _smiles_to_mols(self, query_smiles: List[str]) -> Tuple[List, List]:
        mols = [Chem.MolFromSmiles(smile) for smile in query_smiles]
        valid = [0 if mol is None else 1 for mol in mols]
        valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean == 1]
        valid_mols = [mols[idx] for idx in valid_idxs]
        return valid_mols, valid_idxs
