import os
import multiprocessing as mp

from .base import UNDEFINED_PARAMETER
from .base import (
    TaskBase,
    CollectiveTaskBase
)
from xdalgorithm.toolbox.clustering.inter_molecule_rmsd_clustering_runner import InterMoleculeRMSDClusteringRunner
from xdalgorithm.toolbox.clustering.intra_molecule_dihedral_clustering_runner import IntraMoleculeDihedralClusteringRunner
from xdalgorithm.toolbox.clustering.intra_molecule_rmsd_clustering_runner import IntraMoleculeRMSDClusteringRunner
from xdalgorithm.toolbox.clustering.protein_dihedral_clustering_runner import ProteinDihedralClusteringRunner
from xdalgorithm.engines.utils import merge_dicts_aslist_withref_in_list 

__all__ = [
    "IntraMoleculeDihedralClustering",
    "IntraMoleculeRMSDClustering",
    "InterMoleculeRMSDClustering",
    "ProteinDihedralClustering"
]

class IntraMoleculeDihedralClustering(CollectiveTaskBase):
    """ cluster conformers from the same molecules according the dihedral distance

    Args:
        name (str,optional): the task name. Default to 'intra_molecule_dihedral_clustering'

    Examples:
    >>> data.run_task(
    ...     CollectiveEventBase,
    ...     task=IntraMoleculeDihedralClustering(),
    ...     ligand_sdf_file_name='i:ligand_sdf_file_name:3.autodock',
    ...     ligand_smiles_string='i:ligand_smiles_string:3.autodock',
    ...     cluster_criterion='inconsistent',
    ...     cluster_method='average',
    ...     cluster_cutoff=0.1743
    ... )
    """
    def __init__(self, name='intra_molecule_dihedral_clustering'):
        super().__init__(name)
        self.config_template = {
            'ligand_smiles_string': UNDEFINED_PARAMETER,
            'ligand_sdf_file_name': UNDEFINED_PARAMETER,
            'cluster_method': 'average',
            'cluster_cutoff': 0.1743,
            'cluster_criterion': 'inconsistent',
            'working_dir_name': None
        }

    def collect_config(self, task: TaskBase):
        if isinstance(self.config_template, dict):
            self.cluster_method = self.config_template['cluster_method']
            self.cluster_cutoff = self.config_template['cluster_cutoff']
            self.cluster_criterion = self.config_template['cluster_criterion']
            self.working_dir_name = self.config_template['working_dir_name']
            self.config_template = [self.config_template]
        self.config_template.append(task.config_template)

    def run(self):
        num_nodes = len(self.config_template)
        ligand_sdf_file_name_idx_mapping_dict = {}
        for node_idx in range(num_nodes):
            node_ligand_sdf_file_name = self.config_template[node_idx]['ligand_sdf_file_name']
            ligand_sdf_file_name_idx_mapping_dict[node_ligand_sdf_file_name] = node_idx

        grouped_molecule_info_dict = merge_dicts_aslist_withref_in_list(
            self.config_template,
            ref_keys=['ligand_smiles_string'],
            merge_keys=['ligand_sdf_file_name'],
        )

        grouped_molecule_info_value_list = list(grouped_molecule_info_dict.values())
        num_molecules = len(grouped_molecule_info_value_list)

        manager = mp.Manager()
        grouped_cluster_info_dict_list = manager.list()
        grouped_cluster_info_dict_list.extend([None] * num_molecules)
        clustering_process_list = [None] * num_molecules

        for mol_idx in range(num_molecules):
            ligand_sdf_file_name_list = grouped_molecule_info_value_list[mol_idx]['ligand_sdf_file_name']
            working_dir_name = os.path.join(self.working_dir_name, 'ligand_isomer_' + str(mol_idx))
            clustering_process = mp.Process(target=intra_molecule_dihedral_clustering_process, args=(ligand_sdf_file_name_list,
                                                                                                     self.cluster_method,
                                                                                                     self.cluster_cutoff,
                                                                                                     self.cluster_criterion,
                                                                                                     working_dir_name,
                                                                                                     grouped_cluster_info_dict_list,
                                                                                                     ligand_sdf_file_name_idx_mapping_dict,
                                                                                                     mol_idx))

            clustering_process_list[mol_idx] = clustering_process

        for clustering_process in clustering_process_list:
            clustering_process.start()
        for clustering_process in clustering_process_list:
            clustering_process.join()

        flattened_cluster_info_dict_list = [cluster_info_dict for cluster_info_dict_list in list(grouped_cluster_info_dict_list) for cluster_info_dict in cluster_info_dict_list]

        return flattened_cluster_info_dict_list


class IntraMoleculeRMSDClustering(CollectiveTaskBase):
    """Cluster conformers from the same molecules according the rmsd distance

    Args:
        name (str,optional): the task name. Default to 'intra_molecule_rmsd_clustering'

    Examples:
    ... from xdalgorithm.engines import IntraMoleculeRMSDClustering
    ... from xdalgorithm.engines import CollectiveEventBase
    ... data.run_task(
    ...     CollectiveEventBase,
    ...     task=IntraMoleculeRMSDClustering(),
    ...     protein_conf_name='i:protein_conf_name:0.MODEL.protein_fixer',
    ...     ligand_molecule_name='i:ligand_molecule_name:0.LIGAND.ligand_processor',
    ...     ligand_sdf_file_name='i:ligand_sdf_file_name:0.LIGAND.autodock',
    ...     cluster_method='average',
    ...     cluster_cutoff=1.15,
    ...     cluster_criterion='inconsistent'
    ... )
    """
    def __init__(self, name='intra_molecule_rmsd_clustering'):
        super().__init__(name)
        self.config_template = {
            'protein_conf_name': UNDEFINED_PARAMETER,
            'ligand_molecule_name': UNDEFINED_PARAMETER,
            'ligand_sdf_file_name': UNDEFINED_PARAMETER,
            'cluster_method': 'average',
            'cluster_cutoff': 1.15,
            'cluster_criterion': 'inconsistent',
            'working_dir_name': None
        }
        self.node_idx = 0
        self.mol_idx = 0

    def collect_config(self, task: TaskBase):
        if isinstance(self.config_template, dict):
            self.cluster_method = self.config_template['cluster_method']
            self.cluster_cutoff = self.config_template['cluster_cutoff']
            self.cluster_criterion = self.config_template['cluster_criterion']
            self.working_dir_name = self.config_template['working_dir_name']

            first_group_data_dict = {}
            first_group_data_dict['protein_conf_name'] = self.config_template['protein_conf_name']
            first_group_data_dict['ligand_molecule_name'] = self.config_template['ligand_molecule_name']
            first_group_data_dict['ligand_sdf_file_name_list'] = [self.config_template['ligand_sdf_file_name']]
            first_group_data_dict['node_idx_list'] = [self.node_idx]
            first_group_data_dict['working_dir_name'] = os.path.join(self.working_dir_name, self.config_template['protein_conf_name'] + '_ligand_isomer_' + str(self.mol_idx))

            self.config_template = [first_group_data_dict]

        self.node_idx += 1

        for config in self.config_template:
            existing_group = False
            if task.config_template['ligand_molecule_name'] == config['ligand_molecule_name'] and task.config_template['protein_conf_name'] == config['protein_conf_name']:
                config['ligand_sdf_file_name_list'].append(task.config_template['ligand_sdf_file_name'])
                config['node_idx_list'].append(self.node_idx)
                existing_group = True

        if not existing_group:
            self.mol_idx += 1
            new_group_data_dict = {}
            new_group_data_dict['protein_conf_name'] = task.config_template['protein_conf_name']
            new_group_data_dict['ligand_molecule_name'] = task.config_template['ligand_molecule_name']
            new_group_data_dict['ligand_sdf_file_name_list'] = [task.config_template['ligand_sdf_file_name']]
            new_group_data_dict['node_idx_list'] = [self.node_idx]
            new_group_data_dict['working_dir_name'] = os.path.join(self.working_dir_name, task.config_template['protein_conf_name'] + '_ligand_isomer_' + str(self.mol_idx))

            self.config_template.append(new_group_data_dict)

    def run(self):
        num_molecules = len(self.config_template)
        cluster_info_dict_nested_list = [None] * num_molecules
        for mol_idx in range(num_molecules):
            ligand_sdf_file_name_list = self.config_template[mol_idx]['ligand_sdf_file_name_list']
            source_node_idx_list = self.config_template[mol_idx]['node_idx_list']
            working_dir_name = self.config_template[mol_idx]['working_dir_name']
            os.mkdir(working_dir_name)

            intra_molecule_rmsd_clustering_runner = IntraMoleculeRMSDClusteringRunner(ligand_sdf_file_name_list,
                                                                                      cluster_method=self.cluster_method,
                                                                                      cluster_cutoff=self.cluster_cutoff,
                                                                                      cluster_criterion=self.cluster_criterion,
                                                                                      working_dir_name=working_dir_name)


            _, cluster_info_df = intra_molecule_rmsd_clustering_runner.run()

            num_clusters = cluster_info_df.shape[0]
            cluster_info_dict_list = [None] * num_clusters
            cluster_rank_array = cluster_info_df.index.values

            for cluster_idx in range(num_clusters):
                current_cluster_info_dict = {}
                current_cluster_rank = cluster_rank_array[cluster_idx]
                current_cluster_member_sdf_file_idx_list = cluster_info_df.loc[current_cluster_rank, 'cluster_member_sdf_file_idx']
                current_cluster_info_dict['output_ids'] = [source_node_idx_list[sdf_file_idx] for sdf_file_idx in current_cluster_member_sdf_file_idx_list]
                current_cluster_info_dict['ligand_sdf_file_name'] = cluster_info_df.loc[current_cluster_rank, 'cluster_representative_sdf_file_name']
                current_cluster_info_dict['cluster_rank'] = current_cluster_rank
#                current_cluster_info_dict['cluster_representative_conf_idx'] = cluster_info_df.loc[current_cluster_rank, 'cluster_representative_conf_idx']
#                current_cluster_info_dict['cluster_size'] = cluster_info_df.loc[current_cluster_rank, 'cluster_size']
                cluster_info_dict_list[cluster_idx] = current_cluster_info_dict

            cluster_info_dict_nested_list[mol_idx] = cluster_info_dict_list

        flattened_cluster_info_dict_list = [cluster_info_dict for cluster_info_dict_list in cluster_info_dict_nested_list for cluster_info_dict in cluster_info_dict_list]

        return flattened_cluster_info_dict_list

class InterMoleculeRMSDClustering(CollectiveTaskBase):
    """Cluster conformers from several molecules according the rmsd distance

    Args:
        name (str,optional): the task name. Default to 'inter_molecule_rmsd_clustering'

    Examples:
    >>> data.run_task(
    ...     CollectiveEventBase,
    ...     task=InterMoleculeRMSDClustering(),
    ...     protein_conf_name='i:protein_conf_name:0.MODEL.protein_fixer',
    ...     ligand_sdf_file_name='i:ligand_sdf_file_name:0.LIGAND.intra_molecule_rmsd_clustering',
    ...     core_smarts_string='i:core_smarts_string:0.MODEL.generic_core',
    ...     cluster_method='average',
    ...     cluster_cutoff=1.15,
    ...     cluster_criterion='inconsistent'
    ... )
    """
    def __init__(self, name='inter_molecule_rmsd_clustering'):
        super().__init__(name)
        self.config_template = {
            'protein_conf_name': UNDEFINED_PARAMETER,
            'ligand_sdf_file_name': UNDEFINED_PARAMETER,
            'core_smarts_string': UNDEFINED_PARAMETER,
            'cluster_method': 'average',
            'cluster_cutoff': 1.15,
            'cluster_criterion': 'inconsistent',
            'working_dir_name': None
        }
        self.node_idx = 0
        self.protein_conf_idx = 0

    def collect_config(self, task: TaskBase):
        if isinstance(self.config_template, dict):
            self.core_smarts_string = self.config_template['core_smarts_string']
            self.cluster_method = self.config_template['cluster_method']
            self.cluster_cutoff = self.config_template['cluster_cutoff']
            self.cluster_criterion = self.config_template['cluster_criterion']
            self.working_dir_name = self.config_template['working_dir_name']

            first_group_data_dict = {}
            first_group_data_dict['protein_conf_name'] = self.config_template['protein_conf_name']
            first_group_data_dict['ligand_sdf_file_name_list'] = [self.config_template['ligand_sdf_file_name']]
            first_group_data_dict['node_idx_list'] = [self.node_idx]
            first_group_data_dict['working_dir_name'] = os.path.join(self.working_dir_name, self.config_template['protein_conf_name'])

            self.config_template = [first_group_data_dict]

        self.node_idx += 1

        for config in self.config_template:
            existing_group = False
            if task.config_template['protein_conf_name'] == config['protein_conf_name']:
                config['ligand_sdf_file_name_list'].append(task.config_template['ligand_sdf_file_name'])
                config['node_idx_list'].append(self.node_idx)
                existing_group = True

        if not existing_group:
            self.protein_conf_idx += 1
            new_group_data_dict = {}
            new_group_data_dict['protein_conf_name'] = task.config_template['protein_conf_name']
            new_group_data_dict['ligand_sdf_file_name_list'] = [task.config_template['ligand_sdf_file_name']]
            new_group_data_dict['node_idx_list'] = [self.node_idx]
            new_group_data_dict['working_dir_name'] = os.path.join(self.working_dir_name, task.config_template['protein_conf_name'])

            self.config_template.append(new_group_data_dict)

    def run(self):
        num_protein_conformations = len(self.config_template)
        cluster_info_dict_nested_list = [None] * num_protein_conformations
        for protein_conf_idx in range(num_protein_conformations):
            ligand_sdf_file_name_list = self.config_template[protein_conf_idx]['ligand_sdf_file_name_list']
            source_node_idx_list = self.config_template[protein_conf_idx]['node_idx_list']
            working_dir_name = self.config_template[protein_conf_idx]['working_dir_name']
            os.mkdir(working_dir_name)

            inter_molecule_rmsd_clustering_runner = InterMoleculeRMSDClusteringRunner(ligand_sdf_file_name_list,
                                                                                      self.core_smarts_string,
                                                                                      cluster_method=self.cluster_method,
                                                                                      cluster_cutoff=self.cluster_cutoff,
                                                                                      cluster_criterion=self.cluster_criterion,
                                                                                      working_dir_name=working_dir_name)

            _, cluster_info_df = inter_molecule_rmsd_clustering_runner.run()

            num_clusters = cluster_info_df.shape[0]
            cluster_info_dict_list = [None] * num_clusters
            cluster_rank_array = cluster_info_df.index.values

            for cluster_idx in range(num_clusters):
                current_cluster_info_dict = {}
                current_cluster_rank = cluster_rank_array[cluster_idx]
                current_cluster_member_sdf_file_idx_list = cluster_info_df.loc[current_cluster_rank, 'cluster_member_sdf_file_idx']
                current_cluster_info_dict['output_ids'] = [source_node_idx_list[sdf_file_idx] for sdf_file_idx in current_cluster_member_sdf_file_idx_list]
                current_cluster_info_dict['ligand_sdf_file_name'] = cluster_info_df.loc[current_cluster_rank, 'cluster_representative_sdf_file_name']
                current_cluster_info_dict['cluster_rank'] = current_cluster_rank
#                current_cluster_info_dict['cluster_representative_conf_idx'] = cluster_info_df.loc[current_cluster_rank, 'cluster_representative_conf_idx']
#                current_cluster_info_dict['cluster_size'] = cluster_info_df.loc[current_cluster_rank, 'cluster_size']
                cluster_info_dict_list[cluster_idx] = current_cluster_info_dict

            cluster_info_dict_nested_list[protein_conf_idx] = cluster_info_dict_list

        flattened_cluster_info_dict_list = [cluster_info_dict for cluster_info_dict_list in cluster_info_dict_nested_list for cluster_info_dict in cluster_info_dict_list]

        return flattened_cluster_info_dict_list

class ProteinDihedralClustering(CollectiveTaskBase):
    def __init__(self, name='protein_dihedral_clustering'):
        super().__init__(name)
        self.config_template = {
            'system_solvated_prmtop_file_name': UNDEFINED_PARAMETER,
            'trajectory_dcd_file_name': UNDEFINED_PARAMETER,
            'selection_string': 'byres (protein and around 3.5 resname MOL)',
            'cluster_method': 'average',
            'cluster_cutoff': 0.1743,
            'cluster_criterion': 'inconsistent',
            'working_dir_name': None
        }

    def collect_config(self, task: TaskBase):
        pass

    def run(self):
        protein_dihedral_clustering_runner = ProteinDihedralClusteringRunner(self.config_template['system_solvated_prmtop_file_name'],
                                                                             self.config_template['trajectory_dcd_file_name'],
                                                                             selection_string=self.config_template['selection_string'],
                                                                             cluster_method=self.config_template['cluster_method'],
                                                                             cluster_cutoff=0.1743,
                                                                             cluster_criterion='inconsistent',
                                                                             working_dir_name='.')

        cluster_info_df = protein_dihedral_clustering_runner.run()

        num_clusters = cluster_info_df.shape[0]
        cluster_info_dict_list = [None] * num_clusters
        cluster_rank_array = cluster_info_df.index.values

        for cluster_idx in range(num_clusters):
            current_cluster_rank = cluster_rank_array[cluster_idx]
            current_cluster_info_dict = {}
            current_cluster_info_dict['output_ids'] = cluster_info_df.loc[current_cluster_rank, 'cluster_member_conf_idx']
            current_cluster_info_dict['protein_file_name'] = cluster_info_df.loc[current_cluster_rank, 'cluster_representative_pdb_file_name']
            current_cluster_info_dict['cluster_rank'] = current_cluster_rank
            current_cluster_info_dict['cluster_representative_conf_idx'] = cluster_info_df.loc[current_cluster_rank, 'cluster_representative_conf_idx']
            current_cluster_info_dict['cluster_size'] = cluster_info_df.loc[current_cluster_rank, 'cluster_size']
            cluster_info_dict_list[cluster_idx] = current_cluster_info_dict

        return cluster_info_dict_list
