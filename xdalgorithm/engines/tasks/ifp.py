import os
import numpy as np
from rdkit import Chem

from .base import TaskBase, CollectiveTaskBase
from .base import UNDEFINED_PARAMETER

from xdalgorithm.toolbox.interaction_fingerprints.ifp_calculations_runner import IFPCalculationsRunner
from xdalgorithm.toolbox.interaction_fingerprints.rgroup_ifp_analysis_runner import RGroupIFPAnalysisRunner

__all__ = [
    "IFP",
    "RGroupIFP",
]

class IFP(CollectiveTaskBase):
    def __init__(self, name='ifp'):
        """Calculates Interaction Fingerprints from a binding pose

        Args:
            name (str,optional): the task name. Default to `ifp`

        Examples:
        ... from xdalgorithm.engines import IFP
        ... from xdalgorithm.engines import CollectiveEventBase

        >>> data.run_task(
        ...     CollectiveEventBase,
        ...     task=IFP(),
        ...     protein_pdb_file_name='i:protein_pdb_file_name:0.MODEL.protein_system_builder',
        ...     ligand_smiles_string='i.ligand_smiles_string:0.LIGAND.ligand_processor',
        ...     ligand_sdf_file_name='i:ligand_sdf_file_name:0.LIGAND.autodock'
        ... )
        """
        super().__init__(name=name)
        self.config_template = {
            'protein_pdb_file_name': UNDEFINED_PARAMETER,
            'ligand_smiles_string': UNDEFINED_PARAMETER,
            'ligand_sdf_file_name': UNDEFINED_PARAMETER,
            'ligand_resname': 'MOL',
            'ligand_charge_method': 'gas',
            'include_general_contacts': False,
            'donors_selection_string': 'name F* or name N* or name O*',
            'hydrogens_selection_string': 'name H*',
            'acceptors_selection_string': 'name F* or name N* or name O*',
            'donor_hydrogen_distance_cutoff': 2.0,
            'donor_acceptor_distance_cutoff': 3.6,
            'donor_hydrogen_acceptor_angle_cutoff': 120.0,
            'working_dir_name': None
        }

    def collect_config(self, task: TaskBase):
        if isinstance(self.config_template['protein_pdb_file_name'], str):
            self.config_template['protein_pdb_file_name'] = [self.config_template['protein_pdb_file_name'], task.config_template['protein_pdb_file_name']]
        elif isinstance(self.config_template['protein_pdb_file_name'], list):
            self.config_template['protein_pdb_file_name'].append(task.config_template['protein_pdb_file_name'])
        else:
            raise TypeError('Input protein_pdb_file_name expected a list or a string.')

        if isinstance(self.config_template['ligand_smiles_string'], str):
            self.config_template['ligand_smiles_string'] = [self.config_template['ligand_smiles_string'], task.config_template['ligand_smiles_string']]
        elif isinstance(self.config_template['ligand_smiles_string'], list):
            self.config_template['ligand_smiles_string'].append(task.config_template['ligand_smiles_string'])
        else:
            raise TypeError('Input ligand_smiles_string expected a list or a string.')

        if isinstance(self.config_template['ligand_sdf_file_name'], str):
            self.config_template['ligand_sdf_file_name'] = [self.config_template['ligand_sdf_file_name'], task.config_template['ligand_sdf_file_name']]
        elif isinstance(self.config_template['ligand_sdf_file_name'], list):
            self.config_template['ligand_sdf_file_name'].append(task.config_template['ligand_sdf_file_name'])
        else:
            raise TypeError('Input ligand_sdf_file_name expected a list or a string.')

    def run(self):
        num_source_nodes = len(self.config_template['protein_pdb_file_name'])
        uncollected_data_list = [UNDEFINED_PARAMETER] * num_source_nodes
        if self.config_template['ligand_smiles_string'] == uncollected_data_list:
            self.config_template['ligand_smiles_string'] = [None] * num_source_nodes
            for source_node_idx, ligand_sdf_file_name in enumerate(self.config_template['ligand_sdf_file_name']):
                mol = Chem.SDMolSupplier(ligand_sdf_file_name, removeHs=False)[0]
                self.config_template['ligand_smiles_string'][source_node_idx] = mol.GetProp('smiles_string')

        source_node_list = [None] * num_source_nodes
        for source_node_idx in range(num_source_nodes):
            source_node_list[source_node_idx] = self.config_template['ligand_sdf_file_name'][source_node_idx]

        complex_ligand_sdf_file_name_dict = {}
        complex_ligand_smiles_string_dict = {}

        protein_pdb_file_name_array = np.array(self.config_template['protein_pdb_file_name'], dtype='U')
        ligand_ligand_sdf_file_name_array = np.array(self.config_template['ligand_sdf_file_name'], dtype='U')
        ligand_ligand_smiles_string_array = np.array(self.config_template['ligand_smiles_string'], dtype='U')

        unique_protein_pdb_file_name_array, protein_unique_mapping_array = np.unique(protein_pdb_file_name_array, return_inverse=True)
        num_protein_conformations = unique_protein_pdb_file_name_array.shape[0]

        for protein_conf_idx in range(num_protein_conformations):
            current_mapping_idx_list = np.where(protein_unique_mapping_array == protein_conf_idx)[0].tolist()
            current_protein_pdb_file_name = unique_protein_pdb_file_name_array[protein_conf_idx]
            complex_ligand_sdf_file_name_dict[current_protein_pdb_file_name] = ligand_ligand_sdf_file_name_array[current_mapping_idx_list].tolist()
            complex_ligand_smiles_string_dict[current_protein_pdb_file_name] = ligand_ligand_smiles_string_array[current_mapping_idx_list].tolist()

        ifp_calculations_runner = IFPCalculationsRunner(complex_ligand_sdf_file_name_dict=complex_ligand_sdf_file_name_dict,
                                                        complex_ligand_smiles_string_dict=complex_ligand_smiles_string_dict,
                                                        ligand_resname=self.config_template['ligand_resname'],
                                                        ligand_charge_method=self.config_template['ligand_charge_method'],
                                                        include_general_contacts=self.config_template['include_general_contacts'],
                                                        donors_selection_string=self.config_template['donors_selection_string'],
                                                        hydrogens_selection_string=self.config_template['hydrogens_selection_string'],
                                                        acceptors_selection_string=self.config_template['acceptors_selection_string'],
                                                        donor_hydrogen_distance_cutoff=self.config_template['donor_hydrogen_distance_cutoff'],
                                                        donor_acceptor_distance_cutoff=self.config_template['donor_acceptor_distance_cutoff'],
                                                        donor_hydrogen_acceptor_angle_cutoff=self.config_template['donor_hydrogen_acceptor_angle_cutoff'],
                                                        working_dir_name=self.config_template['working_dir_name'])

        complex_ifp_analysis_summary_df_dict, complex_ligand_sdf_file_name_group_dict, complex_ligand_smiles_string_group_dict = ifp_calculations_runner.run()

        ifp_results_list = []
        for protein_conf_idx in range(num_protein_conformations):
            current_protein_pdb_file_name = unique_protein_pdb_file_name_array[protein_conf_idx]
            current_complex_ifp_analysis_summary_df_list = complex_ifp_analysis_summary_df_dict[current_protein_pdb_file_name]
            current_complex_ligand_sdf_file_name_group_list = complex_ligand_sdf_file_name_group_dict[current_protein_pdb_file_name]
            current_num_molecules = len(current_complex_ifp_analysis_summary_df_list)

            for mol_idx in range(current_num_molecules):
                ifp_analysis_summary_df = current_complex_ifp_analysis_summary_df_list[mol_idx]
                ligand_sdf_file_name_list = current_complex_ligand_sdf_file_name_group_list[mol_idx]
                num_ligand_conformations = ifp_analysis_summary_df.shape[0]

                for conf_idx in range(num_ligand_conformations):
                    ligand_sdf_file_name = ligand_sdf_file_name_list[conf_idx]
                    source_node_idx = source_node_list.index(ligand_sdf_file_name)

                    ifp_results_dict = {}
                    ifp_results_dict['output_ids'] = [source_node_idx]
                    ifp_results_dict['ifp_description_string_list'] = ifp_analysis_summary_df.loc[conf_idx, 'ifp_description_string_list']
#                    ifp_results_dict['test_complex_ligand_sdf_file_name_group_dict'] = complex_ligand_sdf_file_name_group_dict
#                    ifp_results_dict['test_complex_ligand_smiles_string_group_dict'] = complex_ligand_smiles_string_group_dict
#                    ifp_results_dict['num_hydrogen_bond_pairs'] = ifp_analysis_summary_df.loc[conf_idx, 'num_hydrogen_bond_pairs']
                    ifp_results_list.append(ifp_results_dict)

        return ifp_results_list

class RGroupIFP(CollectiveTaskBase):
    def __init__(self, name='rgroup_ifp'):
        """Extracts rgroup ifp data from interaction info.

        Args:
            name (str,optional): the task name. Default to `rgroup_ifp`

        Examples:
        ... from xdalgorithm.engines import RGroupIFP
        ... from xdalgorithm.engines import CollectiveEventBase

        >>> data.run_task(
        ...     CollectiveEventBase,
        ...     task=RGroupIFP(),
        ...     ligand_smiles_string='i:ligand_smiles_string:0.LIGAND.ligand_processor',
        ...     ifp_description_string_list='i:ifp_description_string_list:0.LIGAND.ifp',
        ...     core_smarts_string='i:core_smarts_string:0.MODEL.generic_core'
        ... )
        """
        super().__init__(name=name)
        self.config_template = {
            'ligand_smiles_string': UNDEFINED_PARAMETER,
            'ifp_description_string_list': UNDEFINED_PARAMETER,
            'core_smarts_string': UNDEFINED_PARAMETER,
            'working_dir_name': None
        }

        self.total_ifp_description_string_list = []

    def collect_config(self, task: TaskBase):
        self.core_smarts_string = self.config_template['core_smarts_string']
        self.working_dir_name = self.config_template['working_dir_name']

        if isinstance(self.config_template['ligand_smiles_string'], str):
            self.config_template['ligand_smiles_string'] = [self.config_template['ligand_smiles_string'], task.config_template['ligand_smiles_string']]
        elif isinstance(self.config_template['ligand_smiles_string'], list):
            self.config_template['ligand_smiles_string'].append(task.config_template['ligand_smiles_string'])
        else:
            raise TypeError('Input ligand_smiles_string expected a list or a string.')

        if len(self.total_ifp_description_string_list) == 0:
            self.total_ifp_description_string_list.append(self.config_template['ifp_description_string_list'])

        self.total_ifp_description_string_list.append(task.config_template['ifp_description_string_list'])

    def run(self):
        rgroup_ifp_analysis_runner = RGroupIFPAnalysisRunner(ligand_smiles_string_list=self.config_template['ligand_smiles_string'],
                                                             ifp_description_string_nested_list=self.total_ifp_description_string_list,
                                                             core_smarts_string=self.core_smarts_string)

        rgroup_ifp_vector_group_list, rgroup_ifp_label_array, rgroup_atom_names_dict_group_list, ligand_conf_idx_group_list = rgroup_ifp_analysis_runner.run()
        num_molecules = len(ligand_conf_idx_group_list)

        rgroup_ifp_node_list = []
        for mol_idx in range(num_molecules):
            current_rgroup_ifp_vector_list = rgroup_ifp_vector_group_list[mol_idx]
            current_rgroup_atom_names_dict = rgroup_atom_names_dict_group_list[mol_idx]
            current_ligand_conf_idx_list = ligand_conf_idx_group_list[mol_idx]
            current_num_ligand_conformations = len(current_ligand_conf_idx_list)
            for conf_idx in range(current_num_ligand_conformations):
                rgroup_ifp_vector = current_rgroup_ifp_vector_list[conf_idx]
                source_node_idx = current_ligand_conf_idx_list[conf_idx]

                rgroup_ifp_node_dict = {}
                rgroup_ifp_node_dict['output_ids'] = [source_node_idx]
                rgroup_ifp_node_dict['rgroup_ifp_label_array'] = rgroup_ifp_label_array
                rgroup_ifp_node_dict['rgroup_ifp_vector'] = rgroup_ifp_vector
                rgroup_ifp_node_dict['rgroup_atom_names_dict'] = current_rgroup_atom_names_dict
                rgroup_ifp_node_list.append(rgroup_ifp_node_dict)

        return rgroup_ifp_node_list
