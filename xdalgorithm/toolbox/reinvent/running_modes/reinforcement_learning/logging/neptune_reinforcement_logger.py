import os

import numpy as np
import torch

# Legacy version of Neptune API which is only applied for on-premise installation
# https://docs-legacy.neptune.ai/getting-started/quick-starts/hello-world.html
import neptune
from rdkit import Chem

import xdalgorithm.toolbox.reinvent.utils.logging.reinforcement_learning as ul_rl
from xdalgorithm.toolbox.reinvent.running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from xdalgorithm.toolbox.reinvent.running_modes.reinforcement_learning.logging import ConsoleMessage
from xdalgorithm.toolbox.reinvent.running_modes.reinforcement_learning.logging.base_reinforcement_logger import BaseReinforcementLogger
from xdalgorithm.toolbox.reinvent.scoring.score_summary import FinalSummary
from xdalgorithm.toolbox.reinvent.utils import fraction_valid_smiles,scaffold_num_in_smiles,unique_smiles_num
from xdalgorithm.toolbox.reinvent.utils.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
from xdalgorithm.toolbox.reinvent.utils.logging.neptune import add_mols, add_frequent_scaffolds, draw_umap

class NeptuneReinforcementLogger(BaseReinforcementLogger):
    """Collects stats for reinforcement learning logs."""

    def __init__(self, configuration: GeneralConfigurationEnvelope):
        super().__init__(configuration)
        # if NEPTUNE_API_TOKEN has been defined
        self._exp = None
        if 'NEPTUNE_API_TOKEN' in os.environ:
            neptune.init(project_qualified_name=self._log_config.logging_path)
            self._exp = neptune.create_experiment()
        else:
            raise Exception("cannot extract NEPTUNE_API_TOKEN from environment variables.")

        # _rows and _columns define the shape of the output grid of molecule images in tensorboard.
        self._rows = 4
        self._columns = 4
        self._sample_size = self._rows * self._columns
        self._sf_component_enum = ScoringFunctionComponentNameEnum()
        self._console_message_formatter = ConsoleMessage()

    def log_message(self, message: str):
        self._logger.info(message)

    def timestep_report(self, start_time, n_steps, step, smiles: np.array,
                        mean_score: np.float32, score_summary: FinalSummary, score: np.array,
                        agent_likelihood: torch.tensor, prior_likelihood: torch.tensor,
                        augmented_likelihood: torch.tensor,
                        zero_scores,invalid_num,smiles_num_in_memory,scaffold_num_in_memory):
        message = self._console_message_formatter.create(start_time, n_steps, step, smiles, mean_score, score_summary,
                                                         score, agent_likelihood, prior_likelihood,
                                                         augmented_likelihood)
        self._logger.info(message)
        self._neptune_report(step, smiles, score, score_summary, agent_likelihood, prior_likelihood,
                                 augmented_likelihood,
                                 zero_scores,invalid_num,smiles_num_in_memory,scaffold_num_in_memory)

    def save_final_state(self, agent, scaffold_filter):
        agent.save(os.path.join(self._log_config.resultdir, 'Agent.ckpt'))
        scaffold_filter.save_to_csv(self._log_config.resultdir, self._log_config.job_name)
        # self._summary_writer_summary_writer.close()
        self.log_out_input_configuration()

    def _neptune_report(self, step, smiles, score, score_summary: FinalSummary, agent_likelihood, prior_likelihood,
                            augmented_likelihood,
                            zero_scores,invalid_num,smiles_num_in_memory,scaffold_num_in_memory):
        neptune.log_metric("nll/avg/prior", x=step, y=prior_likelihood.mean())
        neptune.log_metric("nll/avg/augmented", x=step, y=augmented_likelihood.mean())
        neptune.log_metric("nll/avg/agent", x=step, y=agent_likelihood.mean())

        mean_score = np.mean(score)
        for i, log in enumerate(score_summary.profile):
            neptune.log_metric(score_summary.profile[i].name, x=step, y=np.mean(score_summary.profile[i].score))
        
        neptune.log_metric("Average Score", x=step, y=mean_score)
        neptune.log_metric("Fraction valid SMILES", x=step, y=fraction_valid_smiles(smiles))
        neptune.log_metric("Scaffold Number", x=step, y=scaffold_num_in_smiles(smiles))
        neptune.log_metric("Unique SMILES Number", x=step, y=unique_smiles_num(smiles))
        neptune.log_metric("Zero Scoring Molecule Number", x=step, y=zero_scores)
        neptune.log_metric("Invalid SMILES Number", x=step, y=invalid_num)
        neptune.log_metric("SMILES Number in Memory", x=step, y=smiles_num_in_memory)
        neptune.log_metric("Scaffold Number in Memory", x=step, y=scaffold_num_in_memory)

        if step % 10 == 0:
            self._log_out_smiles_sample(smiles, score, step, score_summary)

    def _log_out_smiles_sample(self, smiles, score, step, score_summary: FinalSummary):
        self._visualize_structures(smiles, score, step, score_summary)

    def _visualize_structures(self, smiles, score, step, score_summary: FinalSummary):

        list_of_mols, legends, pattern = self._check_for_invalid_mols_and_create_legends(smiles, score, score_summary)
        try:
            add_mols(neptune, "Molecules from epoch", list_of_mols[:self._sample_size], self._rows,
                     [x for x in legends], global_step=step, size_per_mol=(320, 320), pattern=pattern)
        except Exception as ex:
            print(f"Error in RDKit has occurred in skipping printout for step {step}.")
        try:
            viz_mols = []
            for smi in smiles:
                viz_mol = Chem.MolFromSmiles(smi)
                if viz_mol is None:
                    continue
                else:
                    viz_mols.append(viz_mol)
            add_frequent_scaffolds(neptune,
                                   tag="Frequent scaffolds from epoch",
                                   mols=viz_mols,
                                   top_frequent=self._rows*self._columns,
                                   mols_per_row=self._rows,
                                   global_step=step,
                                   size_per_mol=(320, 320),
                                   pattern=pattern)
        except Exception as ex:
            print(f"Error in RDKit has occurred in add_frequent_scaffolds function, skipping printout for step {step}.")

    def _check_for_invalid_mols_and_create_legends(self, smiles, score, score_summary: FinalSummary):
        smiles = ul_rl.padding_with_invalid_smiles(smiles, self._sample_size)
        list_of_mols, legend = ul_rl.check_for_invalid_mols_and_create_legend(smiles, score, self._sample_size)
        smarts_pattern = self._get_matching_substructure_from_config(score_summary)
        pattern = ul_rl.find_matching_pattern_in_smiles(list_of_mols=list_of_mols, smarts_pattern=smarts_pattern)

        return list_of_mols, legend, pattern

    def _get_matching_substructure_from_config(self, score_summary: FinalSummary):
        smarts_pattern = ""
        for summary_component in score_summary.scaffold_log:
            if summary_component.parameters.component_type == self._sf_component_enum.MATCHING_SUBSTRUCTURE:
                smarts = summary_component.parameters.smiles
                if len(smarts) > 0:
                    smarts_pattern = smarts[0]
        return smarts_pattern

    def _drop_duplicates(self, smiles_list):
        smiles_set = set()
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            canonical_smiles = Chem.MolToSmiles(mol)
            smiles_set.add(canonical_smiles)
        return list([Chem.MolFromSmiles(smi) for smi in smiles_set])

    def _visualize_umap_figure(self, smiles_list_dict):
        legends = []
        mols_groups = []
        for k in smiles_list_dict.keys():
            unique_mols = self._drop_duplicates(smiles_list_dict[k])
            mols_groups.append(unique_mols)
            legends.append("epoch_{0}".format(k))
        if all([len(i) > 0 for i in mols_groups]):
            draw_umap(neptune,
                      tag="UMAP Plot",
                      mols_groups=mols_groups,
                      legends=legends)
