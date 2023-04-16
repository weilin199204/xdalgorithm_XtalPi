import json
import os
from typing import List

# Legacy version of Neptune API which is only applied for on-premise installation
# https://docs-legacy.neptune.ai/getting-started/quick-starts/hello-world.html
import neptune
from rdkit import Chem

from xdalgorithm.toolbox.reinvent.running_modes.configurations.general_configuration_envelope import \
    GeneralConfigurationEnvelope
from xdalgorithm.toolbox.reinvent.running_modes.transfer_learning.logging.base_transfer_learning_logger import \
    BaseTransferLearningLogger
from xdalgorithm.toolbox.reinvent.utils import fraction_valid_smiles
from xdalgorithm.toolbox.reinvent.utils import scaffold_num_in_smiles, unique_smiles_num, scaffold_smarts_matched_num
from xdalgorithm.toolbox.reinvent.utils.logging.neptune import add_mols, add_frequent_scaffolds, draw_umap

class NeptuneTransferLearningLogger(BaseTransferLearningLogger):
    """Collects stats for an existing RNN model."""

    def __init__(self, configuration: GeneralConfigurationEnvelope):
        super().__init__(configuration)
        # if NEPTUNE_API_TOKEN has been defined
        self._exp = None
        if 'NEPTUNE_API_TOKEN' in os.environ:
            neptune.init(project_qualified_name=self._log_config.logging_path)
            self._exp = neptune.create_experiment()
        else:
            raise Exception("cannot extract NEPTUNE_API_TOKEN from environment variables.")

        self.smarts = self._log_config.core_smarts

    def __del__(self):
        if self._exp is not None:
            self._exp.stop()

    def log_out_input_configuration(self):
        file = os.path.join(self._log_config.logging_path, "input.json")
        jsonstr = json.dumps(self._configuration, default=lambda x: x.__dict__, sort_keys=True, indent=4,
                             separators=(',', ': '))
        with open(file, 'w') as f:
            f.write(jsonstr)

    def log_timestep(self, lr, epoch, sampled_smiles, sampled_nlls,
                     validation_nlls, training_nlls, jsd_data, jsd_joined_data, model):
        self.log_message(f"Collecting data for epoch {epoch}")

        # if self._with_weights:
            # self._weight_stats(model, epoch)
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
        neptune.log_metric("lr", x=epoch, y=lr)

    def _valid_stats(self, smiles, epoch):
        neptune.log_metric("valid", x=epoch, y=fraction_valid_smiles(smiles))

    def _scaffold_stats(self, smiles, epoch):
        neptune.log_metric("scaffold number", x=epoch, y=scaffold_num_in_smiles(smiles))

    def _unique_stats(self, smiles, epoch):
        neptune.log_metric("unique smiles num", x=epoch, y=unique_smiles_num(smiles))

    def _matched_scaffold_smarts_counts(self, smiles, epoch):
        if self.smarts is None:
            neptune.log_metric("smiles ratio matched core smarts", x=epoch, y=0)
            neptune.log_metric("unique matched core smarts", x=epoch, y=0)
        elif Chem.MolFromSmarts(self.smarts) is None:
            raise ValueError("A error core_smarts {0} found !".format(self.smarts))
        else:
            matched_ratio, matched_smiles_num = scaffold_smarts_matched_num(smiles, self.smarts)
            neptune.log_metric("smiles ratio matched core smarts", x=epoch, y=matched_ratio)
            neptune.log_metric("unique matched core smarts", x=epoch, y=matched_smiles_num)

    # def _weight_stats(self, model, epoch):
        # for name, weights in model.network.named_parameters():
            # self._summary_writer.add_histogram(f"weights/{name}", weights.clone().cpu().data.numpy(), epoch)

    def _nll_stats_with_validation(self, sampled_nlls, validation_nlls, training_nlls, epoch, jsd_data,
                                   jsd_joined_data):
        neptune.log_metric("nll/avg/sample", x=epoch, y=sampled_nlls.mean())
        neptune.log_metric("nll/avg/validation", x=epoch, y=validation_nlls.mean())
        neptune.log_metric("nll/avg/training", x=epoch, y=training_nlls.mean())

        neptune.log_metric("nll/var/sampled", x=epoch, y=sampled_nlls.var())
        neptune.log_metric("nll/var/validation", x=epoch, y=validation_nlls.var())
        neptune.log_metric("nll/var/training", x=epoch, y=training_nlls.var())

        for key in jsd_data:
            val = jsd_data[key]
            neptune.log_metric("nll_plot/jsd/{}".format(key), x=epoch, y=val)
        neptune.log_metric("nll_plot/jsd_joined", x=epoch, y=jsd_joined_data)

    def _nll_stats(self, sampled_nlls, training_nlls, epoch, jsd_data, jsd_joined_data):
        #TODO: show histogram image on Neptune
        neptune.log_metric("nll/avg/sampled", x=epoch, y=sampled_nlls.mean())
        neptune.log_metric("nll/vag/training", x=epoch, y=training_nlls.mean())

        neptune.log_metric("nll/var/sampled", x=epoch, y=sampled_nlls.var())
        neptune.log_metric("nll/var/training", x=epoch, y=training_nlls.var())

        for key in jsd_data:
            val = jsd_data[key]
            neptune.log_metric("nll_plot/jsd/{}".format(key), x=epoch, y=val)
        neptune.log_metric("nll_plot/jsd_joined", x=epoch, y=jsd_joined_data)

    def _visualize_structures(self, smiles: List[str], epoch: int):
        list_of_labels, list_of_mols = self._count_unique_inchi_keys(smiles)
        if len(list_of_mols) > 0:
            add_mols(neptune, 
                    "Most Frequent Molecules", 
                    list_of_mols, 
                    self._rows,
                    list_of_labels,
                    global_step=epoch)

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
            draw_umap(neptune, tag="UMAP Plot", mols_groups=mols_groups, legends=legends)
