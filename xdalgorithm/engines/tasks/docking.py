import os
import numpy as np
from rdkit import Chem

from .base import TaskBase, CollectiveTaskBase
from .base import UNDEFINED_PARAMETER

from xdalgorithm.toolbox.docking_engines.autogrid_runner import AutoGridRunner
from xdalgorithm.toolbox.docking_engines.autodock_runner import AutoDockRunner

__all__ = [
    "AutoGrid",
    "AutoDock",
]

class AutoGrid(TaskBase):
    def __init__(self, name='autogrid'):
        """ AutoGrid in auto dock

        Args:
            name (str,optional): the task name. Default to 'autogrid'

        Examples:
        ... from xdalgorithm.engines import AutoGridTask
        ... from xdalgorithm.engines import SerialEventBase

        >>> data.run_task(
        ...     SerialEventBase,
        ...     event_type='MODEL',
        ...     task=AutoGrid(),
        ...     protein_pdb_file_name='i:protein_pdb_file_name:0.MODEL.protein_system_builder',
        ...     protein_conf_name='i:protein_conf_name:0.MODEL.protein_fixer',
        ...     target_center=[-9.184, -0.525, -16.676]
        ... )
        """
        super().__init__(name=name)
        self.config_template = {
            'protein_pdb_file_name': UNDEFINED_PARAMETER,
            'protein_conf_name': UNDEFINED_PARAMETER,
            'kept_ligand_resname_list': None,
            'target_center': (0.0, 0.0, 0.0),
            'num_grid_points': (60, 60, 60),
            'grid_spacing': (0.375, 0.375, 0.375),
            'working_dir_name': None
        }

    def run(self):
        autogrid_runner = AutoGridRunner(protein_pdb_file_name=self.config_template['protein_pdb_file_name'],
                                         protein_conf_name=self.config_template['protein_conf_name'],
                                         kept_ligand_resname_list=self.config_template['kept_ligand_resname_list'],
                                         target_center=self.config_template['target_center'],
                                         num_grid_points=self.config_template['num_grid_points'],
                                         grid_spacing=self.config_template['grid_spacing'],
                                         working_dir_name=self.config_template['working_dir_name'])

        protein_docking_grid_summary_file_name = autogrid_runner.run()
        return [{'protein_docking_grid_file_name': protein_docking_grid_summary_file_name}]

class AutoDock(CollectiveTaskBase):
    def __init__(self, name='autodock'):
        """ perform molecular docking for selected molecules by autodock-gpu

        Args:
            name (str,optional): the task name. Default to `autodock`

        two environment variables are required:
        (1) "/data/aidd-server/Modules/AutoDock-GPU-1.3/bin"
        (2) "/data/aidd-server/Modules/amber20/bin"

        Examples:
        ... from xdalgorithm.engines import Autodock
        ... from xdalgorithm.engines import CollectiveEventBase

        >>> data.run_task(
        ...     CollectiveEventBase,
        ...     task=AutoDock(),
        ...     protein_docking_grid_file_name='i:protein_docking_grid_file_name:0.MODEL.autogrid',
        ...     ligand_sdf_file_name='i:ligand_sdf_file_name:0.LIGAND.ligand_conformation_generator',
        ...     num_docking_runs=10
        ... )
        """
        super().__init__(name=name)
        self.config_template = {
            'protein_docking_grid_file_name': UNDEFINED_PARAMETER,
            'ligand_sdf_file_name': UNDEFINED_PARAMETER,
            'num_docking_runs': 10,
            'working_dir_name': None
        }

    def collect_config(self, task: TaskBase):
        if isinstance(self.config_template['protein_docking_grid_file_name'], str):
            self.config_template['protein_docking_grid_file_name'] = [self.config_template['protein_docking_grid_file_name'], task.config_template['protein_docking_grid_file_name']]
        elif isinstance(self.config_template['protein_docking_grid_file_name'], list):
            self.config_template['protein_docking_grid_file_name'].append(task.config_template['protein_docking_grid_file_name'])
        else:
            raise TypeError('Input protein_docking_grid_file_name expected a list or a string.')

        if isinstance(self.config_template['ligand_sdf_file_name'], str):
            self.config_template['ligand_sdf_file_name'] = [self.config_template['ligand_sdf_file_name'], task.config_template['ligand_sdf_file_name']]
        elif isinstance(self.config_template['ligand_sdf_file_name'], list):
            self.config_template['ligand_sdf_file_name'].append(task.config_template['ligand_sdf_file_name'])
        else:
            raise TypeError('Input ligand_sdf_file_name expected a list or a string.')

    def run(self):
        num_source_nodes = len(self.config_template['protein_docking_grid_file_name'])
        source_node_tuple_list = [None] * num_source_nodes
        for source_node_idx in range(num_source_nodes):
            source_node_tuple_list[source_node_idx] = (self.config_template['protein_docking_grid_file_name'][source_node_idx], self.config_template['ligand_sdf_file_name'][source_node_idx])

        protein_docking_grid_file_name_array = np.array(self.config_template['protein_docking_grid_file_name'], dtype='U')
        unique_protein_docking_grid_file_name_array = np.unique(protein_docking_grid_file_name_array)
        num_protein_conformations = unique_protein_docking_grid_file_name_array.shape[0]

        unique_protein_docking_grid_file_name_list = unique_protein_docking_grid_file_name_array.tolist()
        ligand_sdf_file_name_array = np.array(self.config_template['ligand_sdf_file_name'], dtype='U')
        unique_ligand_sdf_file_name_list = np.unique(ligand_sdf_file_name_array).tolist()

        autodock_runner = AutoDockRunner(protein_grid_maps_fld_file_name_list=unique_protein_docking_grid_file_name_list,
                                         ligand_sdf_file_name_list=unique_ligand_sdf_file_name_list,
                                         num_docking_runs=self.config_template['num_docking_runs'],
                                         working_dir_name=self.config_template['working_dir_name'])

        docking_pose_summary_info_list = autodock_runner.run()

        docking_results_nested_list = [None] * num_protein_conformations

        for protein_conf_idx in range(num_protein_conformations):
            current_protein_docking_grid_file_name = unique_protein_docking_grid_file_name_list[protein_conf_idx]
            current_docking_pose_summary_info_df = docking_pose_summary_info_list[protein_conf_idx]
            num_ligand_poses = current_docking_pose_summary_info_df.shape[0]
            docking_results_list = [None] * num_ligand_poses

            for ligand_pose_idx in range(num_ligand_poses):
                current_ligand_original_sdf_file_name = current_docking_pose_summary_info_df.loc[ligand_pose_idx, 'ligand_original_sdf_file_name']
                current_source_node_idx = source_node_tuple_list.index((current_protein_docking_grid_file_name, current_ligand_original_sdf_file_name))

                docking_results_dict = {}
                docking_results_dict['output_ids'] = [current_source_node_idx]
                docking_results_dict['ligand_sdf_file_name'] = current_docking_pose_summary_info_df.loc[ligand_pose_idx, 'ligand_sdf_file_name']
                docking_results_dict['docking_score'] = current_docking_pose_summary_info_df.loc[ligand_pose_idx, 'docking_score']
                docking_results_dict['ligand_efficiency'] = current_docking_pose_summary_info_df.loc[ligand_pose_idx, 'ligand_efficiency']
                docking_results_list[ligand_pose_idx] = docking_results_dict

            docking_results_nested_list[protein_conf_idx] = docking_results_list

        flattened_docking_results_list = [docking_results_dict for docking_results_list in docking_results_nested_list for docking_results_dict in docking_results_list]

        return flattened_docking_results_list
