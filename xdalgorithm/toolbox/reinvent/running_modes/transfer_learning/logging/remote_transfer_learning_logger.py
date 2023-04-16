import numpy as np
import requests

import xdalgorithm.toolbox.reinvent.utils.logging.log as utils_log
import xdalgorithm.toolbox.reinvent.utils as utils_general
from xdalgorithm.toolbox.reinvent.running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from xdalgorithm.toolbox.reinvent.running_modes.transfer_learning.logging.base_transfer_learning_logger import BaseTransferLearningLogger
from xdalgorithm.toolbox.reinvent.utils.logging.visualization import mol_to_png_string


class RemoteTransferLearningLogger(BaseTransferLearningLogger):

    def __init__(self, configuration: GeneralConfigurationEnvelope):
        super().__init__(configuration)
        self._is_dev = utils_log._is_development_environment()

    def log_message(self, message: str):
        self._logger.info(message)

    def _notify_server(self, data, to_address):
        """This is called every time we are posting data to server"""
        try:
            self._logger.warning(f"posting to {to_address}")
            headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
            response = requests.post(to_address, json=data, headers=headers)

            if self._is_dev:
                """logs out the response content only when running a test instance"""
                if response.status_code == requests.codes.ok:
                    self._logger.info(f"SUCCESS: {response.status_code}")
                    self._logger.info(response.content)
                else:
                    self._logger.info(f"PROBLEM: {response.status_code}")
                    self._logger.exception(data, exc_info=False)
        except Exception as t_ex:
            self._logger.exception("Exception occurred", exc_info=True)
            self._logger.exception(f"Attempted posting the following data:")
            self._logger.exception(data, exc_info=False)

    def log_timestep(self, lr, epoch, sampled_smiles, sampled_nlls,
                     validation_nlls, training_nlls, jsd_data, jsd_joined_data, model):

        learning_mean = self._mean_learning_curve_profile(sampled_nlls, training_nlls)
        learning_variation = self._variation_learning_curve_profile(sampled_nlls, training_nlls)
        fraction_valid_smiles = utils_general.fraction_valid_smiles(sampled_smiles)

        scaffold_num = utils_general.scaffold_num_in_smiles(sampled_smiles)
        unique_smiles_num = utils_general.unique_smiles_num(sampled_smiles)

        structures_table = self._visualize_structures(sampled_smiles)
        data = self._assemble_timestep_report(epoch, fraction_valid_smiles,
                                              scaffold_num, unique_smiles_num,
                                              structures_table, learning_mean,
                                              learning_variation, sampled_nlls, training_nlls)
        self._notify_server(data, self._log_config.recipient)

    def _visualize_structures(self, smiles):

        list_of_labels, list_of_mols = self._count_unique_inchi_keys(smiles)
        mol_in_base64_string = mol_to_png_string(list_of_mols, molsPerRow=self._columns, subImgSize=(300, 300),
                                                 legend=list_of_labels)
        return mol_in_base64_string

    def _mean_learning_curve_profile(self, sampled_nlls: np.array, training_nlls: np.array):
        learning_curves = {
            "sampled": float(np.float(sampled_nlls.mean())),
            "training": float(np.float(training_nlls.mean()))
        }
        return learning_curves

    def _variation_learning_curve_profile(self, sampled_nlls: np.array, training_nlls: np.array):
        learning_curves = {
            "sampled": float(np.float(sampled_nlls.var())),
            "training": float(np.float(training_nlls.var()))
        }
        return learning_curves

    def _assemble_timestep_report(self, epoch, fraction_valid_smiles, scaffold_num, unique_smiles_num,
                                  structures_table, learning_mean,
                                  learning_variation, sampled_nlls, training_nlls) -> dict:
        timestep_report = {"epoch": epoch,
                           "fraction_valid_smiles": {"valid": fraction_valid_smiles},
                           "scaffold_num":{"scaffold_num":scaffold_num},
                           "unique_smiles_num":{"unique_smiles_num":unique_smiles_num},
                           "sampled_smiles_distribution": {"negative_log_likelihood": sampled_nlls.tolist()},
                           "training_smiles_distribution": {"negative_log_likelihood": training_nlls.tolist()},
                           "structures": structures_table,
                           "learning_mean": learning_mean,
                           "learning_variation": learning_variation
                           }
        return timestep_report

    def _visualize_umap_figure(self, smiles_list_dict):
        raise NotImplementedError("the method in remote transfer learning logger is not implemented!")
