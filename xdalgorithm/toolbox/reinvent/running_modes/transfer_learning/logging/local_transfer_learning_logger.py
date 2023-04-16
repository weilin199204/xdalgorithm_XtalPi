import json
import os
from typing import List

from torch.utils.tensorboard import SummaryWriter
from rdkit import Chem

from xdalgorithm.toolbox.reinvent.running_modes.configurations.general_configuration_envelope import \
    GeneralConfigurationEnvelope
from xdalgorithm.toolbox.reinvent.running_modes.transfer_learning.logging.base_transfer_learning_logger import \
    BaseTransferLearningLogger
from xdalgorithm.toolbox.reinvent.utils import fraction_valid_smiles
from xdalgorithm.toolbox.reinvent.utils import scaffold_num_in_smiles, unique_smiles_num, scaffold_smarts_matched_num
from xdalgorithm.toolbox.reinvent.utils.logging.tensorboard import add_mols, add_frequent_scaffolds, draw_umap

class LocalTransferLearningLogger(BaseTransferLearningLogger):
    """Collects stats for an existing RNN model."""

    def __init__(self, configuration: GeneralConfigurationEnvelope):
        super().__init__(configuration)
        self._summary_writer = SummaryWriter(log_dir=self._log_config.logging_path)
        self.smarts = self._log_config.core_smarts

    def __del__(self):
        self._summary_writer.close()

    def log_out_input_configuration(self):
        file = os.path.join(self._log_config.logging_path, "input.json")
        jsonstr = json.dumps(self._configuration, default=lambda x: x.__dict__, sort_keys=True, indent=4,
                             separators=(',', ': '))
        with open(file, 'w') as f:
            f.write(jsonstr)

    def log_timestep(self, lr, epoch, sampled_smiles, sampled_nlls,
                     validation_nlls, training_nlls, jsd_data, jsd_joined_data, model):
        self.log_message(f"Collecting data for epoch {epoch}")

        if self._with_weights:
            self._weight_stats(model, epoch)
        if validation_nlls is not None:
            self._nll_stats_with_validation(sampled_nlls, validation_nlls, training_nlls, epoch, jsd_data,
                                            jsd_joined_data)
        elif validation_nlls is None:
            self._nll_stats(sampled_nlls, training_nlls, epoch, jsd_data, jsd_joined_data)
        self._valid_stats(sampled_smiles, epoch)
        self._scaffold_stats(sampled_smiles, epoch)
        self._unique_stats(sampled_smiles, epoch)
        self._matched_scaffold_smarts_counts(sampled_smiles, epoch)

        self._visualize_structures(sampled_smiles, epoch)
        self._summary_writer.add_scalar("lr", lr, epoch)

    def _valid_stats(self, smiles, epoch):
        self._summary_writer.add_scalar("valid", fraction_valid_smiles(smiles), epoch)

    def _scaffold_stats(self, smiles, epoch):
        self._summary_writer.add_scalar("scaffold number", scaffold_num_in_smiles(smiles), epoch)

    def _unique_stats(self, smiles, epoch):
        self._summary_writer.add_scalar("unique smiles num", unique_smiles_num(smiles), epoch)

    def _matched_scaffold_smarts_counts(self, smiles, epoch):
        if self.smarts is None:
            self._summary_writer.add_scalar("smiles ratio matched core smarts", 0, epoch)
            self._summary_writer.add_scalar("unique matched core smarts", 0, epoch)
        elif Chem.MolFromSmarts(self.smarts) is None:
            raise ValueError("A error core_smarts {0} found !".format(self.smarts))
        else:
            matched_ratio, matched_smiles_num = scaffold_smarts_matched_num(smiles, self.smarts)
            self._summary_writer.add_scalar("smiles ratio matched core smarts", matched_ratio, epoch)
            self._summary_writer.add_scalar("unique matched core smarts", matched_smiles_num, epoch)

    def _weight_stats(self, model, epoch):
        for name, weights in model.network.named_parameters():
            self._summary_writer.add_histogram(f"weights/{name}", weights.clone().cpu().data.numpy(), epoch)

    def _nll_stats_with_validation(self, sampled_nlls, validation_nlls, training_nlls, epoch, jsd_data,
                                   jsd_joined_data):
        self._summary_writer.add_histogram("nll_plot/sampled", sampled_nlls, epoch)
        self._summary_writer.add_histogram("nll_plot/validation", validation_nlls, epoch)
        self._summary_writer.add_histogram("nll_plot/training", training_nlls, epoch)

        self._summary_writer.add_scalars("nll/avg", {
            "sampled": sampled_nlls.mean(),
            "validation": validation_nlls.mean(),
            "training": training_nlls.mean()
        }, epoch)

        self._summary_writer.add_scalars("nll/var", {
            "sampled": sampled_nlls.var(),
            "validation": validation_nlls.var(),
            "training": training_nlls.var()
        }, epoch)

        self._summary_writer.add_scalars("nll_plot/jsd", jsd_data, epoch)
        self._summary_writer.add_scalar("nll_plot/jsd_joined", jsd_joined_data, epoch)

    def _nll_stats(self, sampled_nlls, training_nlls, epoch, jsd_data, jsd_joined_data):
        self._summary_writer.add_histogram("nll_plot/sampled", sampled_nlls, epoch)
        self._summary_writer.add_histogram("nll_plot/training", training_nlls, epoch)

        self._summary_writer.add_scalars("nll/avg", {
            "sampled": sampled_nlls.mean(),
            "training": training_nlls.mean()
        }, epoch)

        self._summary_writer.add_scalars("nll/var", {
            "sampled": sampled_nlls.var(),
            "training": training_nlls.var()
        }, epoch)

        self._summary_writer.add_scalars("nll_plot/jsd", jsd_data, epoch)
        self._summary_writer.add_scalar("nll_plot/jsd_joined", jsd_joined_data, epoch)

    def _visualize_structures(self, smiles: List[str], epoch: int):
        list_of_labels, list_of_mols = self._count_unique_inchi_keys(smiles)
        if len(list_of_mols) > 0:
            add_mols(self._summary_writer, "Most Frequent Molecules", list_of_mols, self._rows, list_of_labels,
                     global_step=epoch)
            viz_mols = []
            for smi in smiles:
                viz_mol = Chem.MolFromSmiles(smi)
                if viz_mol is None:
                    continue
                else:
                    viz_mols.append(viz_mol)
            add_frequent_scaffolds(writer=self._summary_writer,
                                   tag="Frequent scaffolds from epoch",
                                   mols=viz_mols,
                                   top_frequent=self._rows * self._columns,
                                   mols_per_row=self._rows,
                                   global_step=epoch)

    def _visualize_umap_figure(self, smiles_list_dict):
        legends = []
        mols_groups = []
        for k in smiles_list_dict.keys():
            _, unique_mols = self._count_unique_inchi_keys(smiles_list_dict[k])
            mols_groups.append(unique_mols)
            legends.append("epoch_{0}".format(k))
        if all([len(i) > 0 for i in mols_groups]):
            draw_umap(writer=self._summary_writer,
                      tag="UMAP Plot",
                      mols_groups=mols_groups,
                      legends=legends)
