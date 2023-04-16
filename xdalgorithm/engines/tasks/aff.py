import typing as t
import os
import random
from collections import defaultdict

import torch
import gpytorch
import numpy as np
import pandas as pd

from xdalgorithm.toolbox.binding_predictor.models import GPRegressionModel
# from xdalgorithm.toolbox.binding_predictor.metrics import loss_functions_factory
from .base import (
    CollectiveTaskBase,
    TaskBase,
    UNDEFINED_PARAMETER
)
from ..utils import (
    merge_dicts_aslist_withref_in_list,
    # ifp_dict_to_dataframe,
    HashableDict
)

__all__ = [
    "QSAR",
    "PredictAff",
]


class QSAR(CollectiveTaskBase):
    """ Affinity predictor model training

        Args:
            name (str,optional): the task name. Default to 'qsar'

        Examples:
        >>> data.run_task(
        ...     CollectiveEventBase,
        ...     task=QSAR(),
        ...     ligand_affinity='i:ligand_affinity:0.TASK.add_ligand',
        ...     cluster_rank='i:cluster_rank:5.TASK.inter_molecule_rmsd_clustering',
        ...     rgroup_ifp_label_array='i:rgroup_ifp_label_array:2.MODEL.rgroup_ifp_labels',
        ...     rgroup_ifp_list='i:rgroup_ifp_list:3.TASK.autodock',
        ...     num_training_iters=10
        ... )
        """
    def __init__(self, name='qsar') -> None:
        super().__init__(name)
        self.working_dir = './'
        self.config_template = {
            'ligand_affinity': UNDEFINED_PARAMETER,
            'cluster_rank': UNDEFINED_PARAMETER,
            'num_training_iters': UNDEFINED_PARAMETER,
            'working_dir_name': UNDEFINED_PARAMETER,
            'rgroup_ifp_list': UNDEFINED_PARAMETER,
            'rgroup_ifp_label_array': UNDEFINED_PARAMETER,
            'lr': 0.1
        }

    def collect_config(self, task: "TaskBase"):
        if isinstance(self.config_template, t.Dict):
            self.working_dir = self.config_template['working_dir_name']
            self.num_training_iters = self.config_template['num_training_iters']
            self.lr = self.config_template['lr']
            self.num_fp_bits = len(self.config_template['rgroup_ifp_label_array'])

            self.config_template = [self.config_template]

        self.config_template.append(task.config_template)

    def run(self):
        
        num_configs = int(len(self.config_template) / 3)
        configs = random.sample(self.config_template, num_configs) 
        configs = self.config_template

        print('start merging')
        merged_dict = merge_dicts_aslist_withref_in_list(
            # self.config_template,
            configs,
            ref_keys=['cluster_rank'],
            merge_keys=[
                'ligand_affinity',
                'rgroup_ifp_list'
            ]
        )
        print('merging done')
        
        rank_dict = defaultdict(list)
        for config_idx, config in enumerate(self.config_template):
            rank_dict[config['cluster_rank']].append(config_idx)
        # rank_dict = {
        #     0: [output_ids],
        #     1: [output_ids],
        #     ...
        #     cluster_id: [output_ids]
        # }
        print('Collecting cluster labels')
        cluster_labels = [
            config['cluster_rank'] for config in merged_dict.keys()
        ]
        # model_dict = {}
        model_path_dict = {}
        model_exp_dict = {}
        
        result_list = []
        for i in cluster_labels:
            print(f"Cluster rank: {str(i)}")
            each_result_dict = {}
            # model_dict[i] = {}
            data_i_key = HashableDict({'cluster_rank': i})
            data_i = merged_dict[data_i_key]
            print(f"Collecting IFP data for cluster {str(i)}")
            feature_df = pd.DataFrame(data_i['rgroup_ifp_list'])  # query the samples owing the label,pd.DataFrame
            ifp_array_list = []
            for ifp_list in data_i['rgroup_ifp_list']:
                ifp_array_single = np.zeros([self.num_fp_bits])
                ifp_array_single[ifp_list] = 1
                ifp_array_list.append(ifp_array_single)
            ifp_array = np.vstack(ifp_array_list)

            ligand_affinity_array = np.array(data_i['ligand_affinity'])
            
            data_ids = np.arange(ifp_array.shape[0])
            np.random.shuffle(data_ids)
            num_train = int(len(data_ids) * 4 / 5)
            
            train_x_np = ifp_array[data_ids[:num_train]]
            train_y_np = ligand_affinity_array[data_ids[:num_train]]
            
            test_x_np = ifp_array[data_ids[num_train:]]
            test_y_np = ligand_affinity_array[data_ids[num_train:]]
            
            train_x = torch.FloatTensor(train_x_np)
            train_y = torch.FloatTensor(train_y_np)

            if train_x.shape[0] == 1:
                print(f"WARNING: 1 data for cluster {str(i)}, no model built.")
                continue
            test_x = torch.FloatTensor(test_x_np)
            test_y = torch.FloatTensor(test_y_np)

            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = GPRegressionModel(
                train_x,
                train_y,
                likelihood
            )
            optim = torch.optim.Adam(
                list(model.parameters())
                + list(likelihood.parameters()),
                lr=self.lr
            )
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                likelihood,
                model
            )
            # train the model
            print(f'Start training GP models for cluster {i}')
            for train_iter in range(self.num_training_iters):
                optim.zero_grad()
                output = model(train_x)
                loss = mll(output, train_y)
                loss.backward()
                print(
                    'Cluster label: %d - Iter %d/%d - Loss: %.3f'
                    % (i, train_iter + 1, self.num_training_iters, loss.item())
                )
                optim.step()
            
            # save the model checkpoints
            model_path = self.working_dir + '/' + str(i) + '.ckpt'
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                },
                model_path
            )
            print(f"Model for cluster {i} saved at {model_path}")
            # model_dict[i]['model_path'] = model_path
            model_path_dict[f'model_{str(i)}'] = model_path

            each_result_dict['model_path'] = model_path
            each_result_dict['cluster_label'] = i
            each_result_dict['output_ids'] = rank_dict[i]
            result_list.append(each_result_dict)
            
            def predict(test_x_np, only_values=True):
                model.eval()
                likelihood.eval()
                test_x = torch.FloatTensor(test_x_np)
                # test_y = torch.FloatTensor(test_y_np)
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    # Make predictions
                    observed_pred = likelihood(model(test_x))

                    mu = observed_pred.mean.numpy()
                    sigma = np.sqrt(observed_pred.variance.numpy())
                if only_values:
                    return mu
                else:
                    return mu, sigma

            from xdalgorithm.toolbox.binding_predictor.metrics import loss_functions_factory
            from xdalgorithm.toolbox.explanation.explanation import SHAP_Explanation

            mu = predict(test_x_np)

            # mu_list = mu.tolist()
            # sigma_list = sigma.tolist()
            # r2 = loss_functions_factory['r2'](test_y_np, mu)
            # mse = loss_functions_factory['mse'](test_y_np, mu)
            # mae = loss_functions_factory['mae'](test_y_np, mu)

            # print(f'cluster: str{i} r2: {r2}, mse: {mse}, mae: {mae}')


            # import pickle
            # from xdalgorithm.utils import get_rand_id
            # pick_id = str(get_rand_id())
            # def save_var(x,file_name):
            #     with open(file_name,'wb') as preader:
            #         pickle.dump(x,preader)
            #     print(file_name)
            # save_var(feature_column_names, "./{0}_feature_column_names.pkl".format(pick_id))
            # save_var(test_x_np, './{0}_test_x_np.pkl'.format(pick_id))

            # exper = SHAP_Explanation(test_x_np, predict, feature_column_names)
            # shap_df = exper.get_shap_df()

            # test_ifp_description_string_list = data_i['ifp_description_string_list'][data_ids[num_train:]]
            # test_rgroup_atom_mapping_list = data_i['rgroup_atom_mapping_dict'][data_ids[num_train:]]
            # test_ligand_sdf_file_name = data_i['ligand_sdf_file_name'][data_ids[num_train:]]
            # test_protein_pdb_file_name = data_i['protein_pdb_file_name'][data_ids[num_train:]]

            # save_var(shap_df, './{0}_shap_df.pkl'.format(pick_id))
            # save_var(test_ifp_description_string_list, './{0}_ifp_description_string_list.pkl'.format(pick_id))
            # save_var(test_rgroup_atom_mapping_list,'./{0}_rgroup_atom_mapping_dict.pkl'.format(pick_id))
            # save_var(test_ligand_sdf_file_name, './{0}_ligand_sdf_file_name.pkl'.format(pick_id))
            # save_var(test_protein_pdb_file_name, './{0}_protein_pdb_file_name.pkl'.format(pick_id))

            # fig_path = os.path.join(self.working_dir, f'shap_{str(i)}')
            # os.makedirs(fig_path, exist_ok=True)


            # output_sdf_path = exper.write_to_parser(
            #                                          shap_df=shap_df,
            #                                          ifp_description_string_list=test_ifp_description_string_list,
            #                                          rgroup_atom_mapping_list=test_rgroup_atom_mapping_list,
            #                                          conformer_sdf_list=test_ligand_sdf_file_name,
            #                                          fig_directory=fig_path,
            #                                          pdb_complex_list=test_protein_pdb_file_name
            #                      )

            # model_exp_dict[f'shap_{str(i)}'] = fig_path

            # suppose to receive some parameters to get explanations
            # explanation = SHAP_Explanation(test_features,prediction_function)
        # model_dict = {
        #     'model_ckpt_path': model_path_dict,
        #     # 'model_exp_path': model_exp_dict
        # }
        return result_list


class PredictAff(CollectiveTaskBase):
    def __init__(self, name: str = 'predict_reports'):
        """[summary]

        Args:
            name (str, optional): [description]. Defaults to 'predict_reports'.
        
        Examples:
        >>> data.run_task(
        ...     CollectiveEventBase,
        ...     task=PredictAff(),
        ...     ligand_molecule_name='i:ligand_molecule_name:0.TASK.add_ligand',
        ...     ligand_affinity='i:ligand_affinity:0.TASK.add_ligand',
        ...     cluster_rank='i:cluster_rank:5.TASK.inter_molecule_rmsd_clustering',
        ...     model_path='i:model_path:6.TASK.qsar',
        ...     rgroup_ifp_list='i:rgroup_ifp_list:3.TASK.autodock',
        ...     rgroup_ifp_label_array='i:rgroup_ifp_label_array:2.MODEL.add_rgroup_ifp_labels'
        ... )
        """
        super().__init__(name)
        self.working_dir = './'
        self.config_template = {
            'ligand_molecule_name': UNDEFINED_PARAMETER,
            'ligand_affinity': UNDEFINED_PARAMETER,
            'cluster_rank': UNDEFINED_PARAMETER,
            'model_path': UNDEFINED_PARAMETER,
            'rgroup_ifp_list': UNDEFINED_PARAMETER,
            'rgroup_ifp_label_array': UNDEFINED_PARAMETER,
            'working_dir_name': UNDEFINED_PARAMETER
        }

    def collect_config(self, task: "TaskBase"):
        if isinstance(self.config_template, t.Dict):
            self.working_dir = self.config_template['working_dir_name']
            # self.model_ckpt_dict = self.config_template['model_ckpt_path']  # t.Dict
            # self.num_training_iters = self.config_template['num_training_iters']
            self.num_fp_bits = len(self.config_template['rgroup_ifp_label_array'])
            self.config_template = [self.config_template]

        self.config_template.append(task.config_template)

    def run(self):
        print("Merging config templates") 
        merged_dict = merge_dicts_aslist_withref_in_list(
            self.config_template,
            ref_keys=['cluster_rank'],
            merge_keys=[
                'rgroup_ifp_list',
                'ligand_affinity',
                'ligand_molecule_name',
                'model_path',
            ]
        )
        print("Merged! hahahaha")

        cluster_labels = [
            config['cluster_rank'] for config in merged_dict.keys()
        ]  # t.List[int]
        print(f"Cluster labels are {cluster_labels}")

        print("Collecting cluster labels...")
        cluster_modelpath_dict = {}
        for cluster_label in cluster_labels:
            cluster_key = HashableDict({'cluster_rank': cluster_label})
            cluster_modelpath_dict[cluster_label] = merged_dict[cluster_key][
                'model_path'
            ][0]
        # cluster_modelpath_dict = {
        #     cluster_label(int): model_path(str),
        #     ...
        # }
        
        model_dict = {}
        likelihood_dict = {}

        # model_name = f'model_{str(cluster_label)}' 
        for cluster_label, model_path in cluster_modelpath_dict.items():
            print(f"Using model for label {cluster_label}")
            checkpoint = torch.load(model_path)

            likelihood_dict[cluster_label] = gpytorch.likelihoods.GaussianLikelihood()

            random_x, random_y = [], []
            for value in merged_dict.values():
                ifp_list = []
                for ifp_array in value['rgroup_ifp_list']:
                    ifp_array_i = np.zeros([self.num_fp_bits])
                    ifp_array_i[ifp_array] = 1
                    ifp_list.append(ifp_array_i)
                random_x = torch.FloatTensor(
                    np.vstack(ifp_list)
                )
                random_y = torch.FloatTensor(
                    np.array(value['ligand_affinity'])
                )
                break

            model_dict[cluster_label] = GPRegressionModel(
                random_x,
                random_y,
                likelihood_dict[cluster_label],
            )
            model_dict[cluster_label].load_state_dict(checkpoint['model_state_dict'])

        result_dict = {}
        result_dict['predicted_reports'] = {}
        predicted_reports = result_dict['predicted_reports']
        
        # mol_name_list = list(set(self.config_template['ligand_molecule_name']))
        mol_name_list = [config['ligand_molecule_name'] for config in self.config_template]
        mol_name_list = list(set(mol_name_list))

        for mol_name in mol_name_list:
            predicted_reports[mol_name] = {}
            predicted_reports[mol_name]['aff'] = 0
            predicted_reports[mol_name]['predicted'] = {}

        for i in cluster_labels:
            # model_name = f'model_{str(i)}'
            # if model_name not in model_dict.keys():
            #     break
            model = model_dict[i]
            likelihood = likelihood_dict[i]
            model.eval()
            likelihood.eval()

            data_i_key = HashableDict({'cluster_rank': i})
            data_i = merged_dict[data_i_key]

            # TODO: 要改
            ifp_list = []
            for ifp_array in data_i['rgroup_ifp_list']:
                ifp_array_i = np.zeros([self.num_fp_bits])
                ifp_array_i[ifp_array] = 1
                ifp_list.append(ifp_array_i)
            data_i['ifp_array'] = np.vstack(ifp_list)
            data_i['ligand_affinity_array'] = np.array(data_i['ligand_affinity'])

            x = torch.FloatTensor(data_i['ifp_array'])

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                # Make predictions
                observed_pred = likelihood(model(x))

                mu = observed_pred.mean.numpy()
                sigma = np.sqrt(observed_pred.variance.numpy())
            
            data_i['aff_pred'] = mu
            data_i['aff_sigma'] = sigma

            for mol_name, aff, aff_pred, aff_sigma in zip(
                data_i['ligand_molecule_name'],
                data_i['ligand_affinity'],
                data_i['aff_pred'],
                data_i['aff_sigma'],
            ):
                predicted_reports[mol_name]['aff'] = aff
                predicted_reports[mol_name]['predicted'][i] = [aff_pred, aff_sigma]
            
        return [result_dict]
        # result_dict = {
        #     'predicted_reports': {
        #         'mol_name_1': {
        #             'aff': 0.6,
        #             'predited': {
        #                 0: [mu, sigma],
        #                 1: [mu, sigma],
        #                 ...
        #             }
        #         },
        #         'mol_name_2': {
        #             'aff': 0.7,
        #             'predited': {
        #                 2: [mu, sigma],
        #                 4: [mu, sigma],
        #                 ...
        #             }
        #         },
        #         ...
        #     }
        # }

             
            

            

       


         

