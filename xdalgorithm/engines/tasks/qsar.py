import typing as t
import os
import numpy as np
from xdalgorithm.toolbox.binding_predictor.predictors import QSARGPPredictor, QSARRFPredictor
from xdalgorithm.toolbox.explanation import SHAP_Explanation

from .base import (
    CollectiveTaskBase,
    TaskBase,
    UNDEFINED_PARAMETER
)
from ..utils import (
    HashableDict
)


class QSAR(CollectiveTaskBase):
    def __init__(self, name='sqar') -> None:
        super().__init__(name)
        self.config_template = {
            'ligand_affinity': UNDEFINED_PARAMETER,
            'cluster_rank': UNDEFINED_PARAMETER,
            'num_training_iters': UNDEFINED_PARAMETER,
            'working_dir_name': UNDEFINED_PARAMETER,
            'rgroup_ifp_list': UNDEFINED_PARAMETER,
            'rgroup_ifp_label_array': UNDEFINED_PARAMETER,
            # model parameters
            'lr': 0.1,
            'history_model_config': None,
            'predictor_type': 'sqar_rf_predictor',
            'likelihood_type': 'gaussian_likelihood',
            'n_estimators': 100,
            'max_depth': None,
            'n_jobs': None,
            'n_splits': 5,
            'shuffle': True,
            'run_explanation': True,
            # TODO: get feature names
            'feature_name': None,
            'explanation_type': 'shap',
            'sample_num_for_explanation': 5
        }
        self._predictor = None

    def collect_config(self, task: "TaskBase"):
        if isinstance(self.config_template, t.Dict):
            self.working_dir = self.config_template['working_dir_name']
            self.num_training_iters = self.config_template['num_training_iters']
            self.lr = self.config_template['lr']
            self.num_fp_bits = len(self.config_template['rgroup_ifp_label_array'])
            self.config_template = [self.config_template]
        self.config_template.append(task.config_template)

    def run(self):
        # num_configs = int(len(self.config_template) / 3)
        # configs = random.sample(self.config_template, num_configs)
        # configs = self.config_template
        # print('start merging')
        # merged_dict = merge_dicts_aslist_withref_in_list(
        #    configs,
        #    ref_keys=['cluster_rank'],
        #    merge_keys=[
        #        'ligand_affinity',
        #        'rgroup_ifp_list'
        #    ]
        # )
        # print('merging done')

        # rank_dict = defaultdict(list)
        # for config_idx, config in enumerate(self.config_template):
        #    rank_dict[config['cluster_rank']].append(config_idx)
        # print('Collecting cluster labels')
        # cluster_labels = [

        #    config['cluster_rank'] for config in merged_dict.keys()
        # ]

        # result_list = []
        # for i in cluster_labels:
        ################# Prepare data ##########################################
        print(f"Cluster rank: {str(i)}")
        # model_dict[i] = {}
        data_i_key = HashableDict({'cluster_rank': i})
        data_i = merged_dict[data_i_key]
        print(f"Collecting IFP data for cluster {str(i)}")
        ifp_array_list = []
        for ifp_list in data_i['rgroup_ifp_list']:
            ifp_array_single = np.zeros([self.num_fp_bits])
            ifp_array_single[ifp_list] = 1
            ifp_array_list.append(ifp_array_single)
        # Attention a feature array and a affinity array.
        ifp_array = np.vstack(ifp_array_list)
        ligand_affinity_array = np.array(data_i['ligand_affinity'])

        ################# Train a model ############################################
        model_config = self._prepare_training_config(ifp_array, ligand_affinity_array)
        model_result_dict = self._train_new_predictor(model_config)

        ################# Train a model ############################################
        if self.config_template['run_explanation']:
            data_ids = np.arange(ifp_array.shape[0])
            np.random.shuffle(data_ids)
            explanation_sample_indices = data_ids[:self.config_template['sample_num_for_explanation']]
            test_x_np = ifp_array[explanation_sample_indices, :]
            test_ifp_description_string_list = data_i['ifp_description_string_list'][explanation_sample_indices]
            test_rgroup_atom_mapping_list = data_i['rgroup_atom_mapping_dict'][explanation_sample_indices]
            test_ligand_sdf_file_name = data_i['ligand_sdf_file_name'][explanation_sample_indices]
            test_protein_pdb_file_name = data_i['protein_pdb_file_name'][explanation_sample_indices]

            visual_result_dict = self._run_explanation(test_x=test_x_np,
                                                       test_ifp_description_string_list=test_ifp_description_string_list,
                                                       test_rgroup_atom_mapping_list=test_rgroup_atom_mapping_list,
                                                       test_ligand_sdf_file_name=test_ligand_sdf_file_name,
                                                       fig_path=None,
                                                       test_protein_pdb_file_name=test_protein_pdb_file_name)
        else:
            visual_result_dict = None

        node_dict = dict()
        node_dict.update(model_result_dict)
        if visual_result_dict is not None:
            node_dict.update(visual_result_dict)
        # result_list.append(node_dict)
        # return result_list

    def _prepare_training_config(self, ifp_array, ligand_affinity_array):
        if self.config_template['predictor_type'] == 'sqar_rf_predictor':
            model_config = dict(new_x=ifp_array,
                                new_y=ligand_affinity_array,
                                n_estimators=self.config_template['n_estimators'],
                                max_depth=self.config_template['max_depth'],
                                n_jobs=self.config_template.get('n_jobs', 4),
                                use_cross_validation=self.config_template.get('use_cross_validation', False),
                                n_splits=self.config_template.get('n_splits', 5),
                                cross_validation_shuffle=self.config_template.get('cross_validation_shuffle', True),
                                work_dir=self.config_template['working_dir_name']
                                )
            if self.config_template['history_model_config'] is not None:
                model_config['model_path'] = self.config_template['history_model_config']['model_path']
                model_config['history_x_path'] = self.config_template['history_model_config']['train_x_path']
                model_config['history_y_path'] = self.config_template['history_model_config']['train_y_path']
        elif self.config_template['predictor_type'] == 'sqar_gp_predictor':
            model_config = dict(new_x=ifp_array,
                                new_y=ligand_affinity_array,
                                train_steps=self.config_template.get('train_steps', 20),
                                lr=self.config_template.get('lr', 0.1),
                                likelihood_type=self.config_template.get('likelihood_type', 'gaussian_likelihood'),
                                work_dir=self.config_template['working_dir_name']
                                )
            if self.config_template['history_model_config'] is not None:
                model_config['model_path'] = self.config_template['history_model_config']['model_path']
                model_config['history_x_path'] = self.config_template['history_model_config']['train_x_path']
                model_config['history_y_path'] = self.config_template['history_model_config']['train_y_path']
        else:
            raise ValueError('Unknown predictor_type {0}.'.format(self.config_template['predictor_type']))
        return model_config

    def _train_new_predictor(self, predictor_config):
        predictor_type = predictor_config.get('predictor_type', ValueError())
        if predictor_type == 'sqar_rf_predictor':
            self._predictor = QSARRFPredictor(**predictor_config)
        elif predictor_type == 'sqar_gp_predictor':
            self._predictor = QSARGPPredictor(**predictor_config)
        else:
            raise ValueError('Unknown predictor_config, {0} is not found.'.format(predictor_config['predictor_type']))
        result_dict = self._predictor.run()
        return result_dict

    def _run_explanation(self, test_x, test_ifp_description_string_list, test_rgroup_atom_mapping_list,
                         test_ligand_sdf_file_name, fig_path, test_protein_pdb_file_name):
        predict_fun = self._predictor.predict

        if self.config_template['explanation'] == 'shap':
            explainer = SHAP_Explanation(test_x, predict_fun, self.config_template['feature_name'])
            shap_df = explainer.get_shap_df()
            explainer.write_to_parser(shap_df=shap_df,
                                      ifp_description_string_list=test_ifp_description_string_list,
                                      rgroup_atom_mapping_list=test_rgroup_atom_mapping_list,
                                      conformer_sdf_list=test_ligand_sdf_file_name,
                                      fig_directory=fig_path,
                                      pdb_complex_list=test_protein_pdb_file_name
                                      )
            visual_result_config = {
                'pharm_model_path': os.path.join(fig_path, 'pharm-model.csv'),
                'protein_pdb_path': os.path.join(fig_path, 'merged_protein.pdb')
            }
            return visual_result_config
        else:
            raise ValueError('Unknown explanation')
