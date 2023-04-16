import re
import os
import numpy as np
import pandas as pd

import MDAnalysis as mda
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdRGroupDecomposition

from xdalgorithm.toolbox.protein_fixer.protein_fixer_runner import ProteinFixerRunner
from xdalgorithm.toolbox.ligand_preparation.ligand_processor_runner import LigandProcessorRunner
from xdalgorithm.toolbox.ligand_preparation.ligand_conformation_generator_runner import LigandConformationGeneratorRunner
from xdalgorithm.toolbox.md.system_building.protein_system_builder_runner import ProteinSystemBuilderRunner
from xdalgorithm.toolbox.scaffold_network.get_scaffold_net import get_scaffold_network

from .base import UNDEFINED_PARAMETER
from .base import TaskBase
from .base import CollectiveTaskBase

__all__ = [
    "AddLigands",
    "LigandProcessor",
    "LigandConformationGenerator",
    "ProteinFixer",
    "ProteinSystemBuilder",
    "ScaffoldNetwork",
    "Core",
    "GenericCore",
    "RGroupIFPLabels"
]


class AddLigands(TaskBase):
    def __init__(
            self,
            name: str = 'add_ligands'
    ):
        """A task to add ligand info to `Dataset`

        Args:
            name (str, optional): task_name. Defaults to 'add_ligands'.

        Exmaples:
        >>> from xdalgorithm.engines import get_dataset
        ... from xdalgorithm.engines import AddLigands
        ... from xdalgorithm.engines import SerialEventBase

        >>> data=get_dataset()

        >>> data.run_task(
        ...     SerialEventBase,
        ...     task=AddLigands(),
        ...     dataset_csv_file_name='datasets/Miransertib_AKT.csv',
        ... )
        """
        super().__init__(name)
        self.config_template = {'dataset_csv_file_name': UNDEFINED_PARAMETER}

    def run(self):
        dataset_info_df = pd.read_csv(self.config_template['dataset_csv_file_name'])
        num_ligand_molecules = dataset_info_df.shape[0]
        ligand_smiles_string_array = dataset_info_df.loc[:, 'SMILES'].values.astype('U')
        ligand_affinity_array = dataset_info_df.loc[:, 'pX'].values.astype(np.float32)

        if 'ligand_molecule_name' in dataset_info_df.columns:
            ligand_molecule_name_array = dataset_info_df.loc[:, 'ligand_molecule_name'].values.astype('U')
        else:
            ligand_molecule_name_list = [None] * num_ligand_molecules
            for mol_idx in range(num_ligand_molecules):
                ligand_molecule_name_list[mol_idx] = 'ligand_' + str(mol_idx)

            ligand_molecule_name_array = np.array(ligand_molecule_name_list, dtype='U')

        ligand_info_list = [None] * num_ligand_molecules
        for mol_idx in range(num_ligand_molecules):
            dataset_info_dict = {}
            dataset_info_dict['ligand_molecule_name'] = ligand_molecule_name_array[mol_idx]
            dataset_info_dict['ligand_smiles_string'] = ligand_smiles_string_array[mol_idx]
            dataset_info_dict['ligand_affinity'] = ligand_affinity_array[mol_idx]
            ligand_info_list[mol_idx] = dataset_info_dict

        return ligand_info_list

class LigandProcessor(TaskBase):
    def __init__(self, name='ligand_processor'):
        """A task to process ligands and do stereoisomer enumerations 

        Args:
            name (str, optional): the task name. Defaults to "ligand_processor".

        Exmaples:
        >>> from xdalgorithm.engines import get_dataset
        ... from xdalgorithm.engines import LigandProcessor
        ... from xdalgorithm.engines import SerialEventBase

        >>> data=get_dataset()

        >>> data.run_task(
        ...     SerialEventBase,
        ...     task=LigandProcessor(),
        ...     ligand_smiles_string='i:ligand_smiles_string:0.LIGAND.add_ligands',
        ...     ligand_molecule_name='i:ligand_molecule_name:0.LIGAND.add_ligands',
        ... )
        """
        super().__init__(name=name)
        self.config_template = {
            'ligand_smiles_string': UNDEFINED_PARAMETER,
            'ligand_molecule_name': UNDEFINED_PARAMETER,
            'core_filter_smiles_string': None,
            'core_filter_smarts_string': None,
            'process_isomer': 'enumerate',
            'max_num_unspecified_chiral_centers': 3,
            'protonation': True,
            'use_chemaxon': False,
            'working_dir_name': None
        }

    def run(self):
        ligand_processor_runner = LigandProcessorRunner(ligand_smiles_string=self.config_template['ligand_smiles_string'],
                                                        ligand_molecule_name=self.config_template['ligand_molecule_name'],
                                                        core_filter_smiles_string=self.config_template['core_filter_smiles_string'],
                                                        core_filter_smarts_string=self.config_template['core_filter_smarts_string'],
                                                        process_isomer=self.config_template['process_isomer'],
                                                        max_num_unspecified_chiral_centers=self.config_template['max_num_unspecified_chiral_centers'],
                                                        protonation=self.config_template['protonation'],
                                                        use_chemaxon=self.config_template['use_chemaxon'])

        isomer_smiles_string_list, isomer_molecule_name_list = ligand_processor_runner.run()

        num_isomers = len(isomer_smiles_string_list)
        ligand_isomer_nodes_list = [None] * num_isomers
        for isomer_idx in range(num_isomers):
            ligand_isomer_node_info_dict = {}
            ligand_isomer_node_info_dict['ligand_smiles_string'] = isomer_smiles_string_list[isomer_idx]
            ligand_isomer_node_info_dict['ligand_molecule_name'] = isomer_molecule_name_list[isomer_idx]
            ligand_isomer_nodes_list[isomer_idx] = ligand_isomer_node_info_dict

        return ligand_isomer_nodes_list

class LigandConformationGenerator(TaskBase):
    def __init__(self, name='ligand_conformation_generator'):
        """A task to generate ligand conformations

        Args:
            name (str, optional): the task name. Defaults to "ligand_conformation_generatotr".

        Exmaples:
        >>> from xdalgorithm.engines import get_dataset
        ... from xdalgorithm.engines import LigandConformationGenerator
        ... from xdalgorithm.engines import SerialEventBase

        >>> data=get_dataset()

        >>> data.run_task(
        ...     SerialEventBase,
        ...     task=LigandConformationGenerator(),
        ...     ligand_smiles_string='i:ligand_smiles_string:0.LIGAND.ligand_processor',
        ...     ligand_molecule_name='i:ligand_molecule_name:0.LIGAND.ligand_processor',
        ...     n_cpu=60
        ... )
        """
        super().__init__(name=name)
        self.config_template = {
            'ligand_smiles_string': UNDEFINED_PARAMETER,
            'ligand_molecule_name': UNDEFINED_PARAMETER,
            'n_cpu': 8,
            'max_num_conformations_per_isomer': None,
            'max_attempts': None,
            'use_random_coords': True,
            'prune_rmsd_threshold': 0.5,
            'enforce_chirality': True,
            'use_exp_torsion_angle_preferences': True,
            'use_basic_knowledge': True,
            'use_small_ring_torsions': True,
            'remove_twisted_six_rings': True,
            'working_dir_name': None
        }

    def run(self):
        ligand_conformation_generator_runner = LigandConformationGeneratorRunner(ligand_smiles_string=self.config_template['ligand_smiles_string'],
                                                                                 ligand_molecule_name=self.config_template['ligand_molecule_name'],
                                                                                 n_cpu=self.config_template['n_cpu'],
                                                                                 max_num_conformations_per_isomer=self.config_template['max_num_conformations_per_isomer'],
                                                                                 max_attempts=self.config_template['max_attempts'],
                                                                                 use_random_coords=self.config_template['use_random_coords'],
                                                                                 prune_rmsd_threshold=self.config_template['prune_rmsd_threshold'],
                                                                                 enforce_chirality=self.config_template['enforce_chirality'],
                                                                                 use_exp_torsion_angle_preferences=self.config_template['use_exp_torsion_angle_preferences'],
                                                                                 use_basic_knowledge=self.config_template['use_basic_knowledge'],
                                                                                 use_small_ring_torsions=self.config_template['use_small_ring_torsions'],
                                                                                 remove_twisted_six_rings=self.config_template['remove_twisted_six_rings'],
                                                                                 working_dir_name=self.config_template['working_dir_name'])

        ligand_molecule_name_list, ligand_smiles_string_list, ligand_sdf_file_name_list = ligand_conformation_generator_runner.run()
        num_conformations = len(ligand_sdf_file_name_list)

        if num_conformations == 0:
            raise Exception('Reasonable molecular conformation cannot be generated.')
        else:
            ligand_conformation_nodes_list = [None] * num_conformations
            for conf_idx in range(num_conformations):
                ligand_conformation_node_info_dict = {}
                # ligand_conformation_node_info_dict['ligand_molecule_name'] = ligand_molecule_name_list[conf_idx]
                # ligand_conformation_node_info_dict['ligand_smiles_string'] = ligand_smiles_string_list[conf_idx]
                ligand_conformation_node_info_dict['ligand_sdf_file_name'] = ligand_sdf_file_name_list[conf_idx]
                ligand_conformation_nodes_list[conf_idx] = ligand_conformation_node_info_dict

            return ligand_conformation_nodes_list

def run_for_multiple(
    protein_pdb_file_name,
    alignment_option,
    keep_water,
    fill_ter_residue,
    cap_residue,
    long_loop,
    patch_ter,
    num_models,
    removed_chain_id_list,
    kept_ligand_resname_list,
    reference_pdb_file_name_for_alignment,
    working_dir_name 
):

    protein_fixer_runner = ProteinFixerRunner(
        protein_pdb_file_name,
        alignment_option,
        keep_water,
        fill_ter_residue,
        cap_residue,
        long_loop,
        patch_ter,
        num_models,
        removed_chain_id_list,
        kept_ligand_resname_list,
        reference_pdb_file_name_for_alignment,
        working_dir_name
    )
    protein_pdb_file_name_list, protein_conf_name_list = protein_fixer_runner.run()
    return protein_pdb_file_name_list[0], protein_conf_name_list[0]

def run_multiple(config):
    return run_for_multiple(*config)

class ProteinFixer(TaskBase):
    def __init__(self, name='protein_fixer'):
        """Protein fixing task

        Args:
            name (str, optional): the task name. Defaults to 'protein_fixer'.

        Examples:
        >>> from xdalgorithm.engines import get_dataset
        ... from xdalgorithm.engines import ProteinFixer
        ... from xdalgorithm.engines import SerialEventBase

        >>> data=get_dataset()

        >>> data.run_task(
        ...     SerialEventBase, 
        ...     task=ProteinFixer(), 
        ...     protein_pdb_file_name='./datasets/5kcv.pdb',
        ...     event_type='MODEL',
        ...     num_models=2
        ... )
        """
        super().__init__(name=name)
        self.config_template = {
            'alignment_option': 'all',
            'protein_pdb_file_name': UNDEFINED_PARAMETER,
            'keep_water': "none",
            'fill_ter_residue': False,
            'cap_residue': False,
            'long_loop': 5,
            'patch_ter': False,
            'num_models': 1,
            'removed_chain_id_list': None,
            'add_h': False,
            'aligned_pdb_file_name': None,
            'kept_ligand_resname_list': None,
            'working_dir_name': None,
            'reference_pdb_file_name_for_alignment': None
        }

    def run(self):

        self.config_template['protein_pdb_file_name'] = \
            os.path.abspath(self.config_template['protein_pdb_file_name'])

        import multiprocess as mp

        configs = []
        for i in range(self.config_template['num_models']):
            working_dir = os.path.join(
                self.config_template['working_dir_name'],
                f"conf_{str(i)}"
            )
            os.makedirs(working_dir, exist_ok=True)

            config = [
                self.config_template['protein_pdb_file_name'],
                self.config_template['alignment_option'],
                self.config_template['keep_water'],
                self.config_template['fill_ter_residue'],
                self.config_template['cap_residue'],
                self.config_template['long_loop'],
                self.config_template['patch_ter'],
                1,
                self.config_template['removed_chain_id_list'],
                self.config_template['kept_ligand_resname_list'],
                self.config_template['reference_pdb_file_name_for_alignment'],
                working_dir
            ]
            configs.append(config)

        pool = mp.Pool(mp.cpu_count())
        print(len(configs))
        print(configs[0])
        results = pool.map(run_multiple, configs)
        # pool.close()
        # pool.join()
        protein_pdb_file_name_list, protein_conf_name_list = [], []
        for file_name, conf_name in results:
            protein_pdb_file_name_list.append(file_name) 
            protein_conf_name_list.append(conf_name)

        # protein_fixer_runner = ProteinFixerRunner(protein_pdb_file_name=self.config_template['protein_pdb_file_name'],
        #                                           keep_water=self.config_template['keep_water'],
        #                                           fill_ter_residue=self.config_template['fill_ter_residue'],
        #                                           cap_residue=self.config_template['cap_residue'],
        #                                           long_loop=self.config_template['long_loop'],
        #                                           patch_ter=self.config_template['patch_ter'],
        #                                           num_models=self.config_template['num_models'],
        #                                           removed_chain_id_list=self.config_template['removed_chain_id_list'],
        #                                           add_h=self.config_template['add_h'],
        #                                           aligned_protein_pdb_file_name=self.config_template['aligned_pdb_file_name'],
        #                                           kept_ligand_resname_list=self.config_template['kept_ligand_resname_list'],
        #                                           working_dir_name=self.config_template['working_dir_name'])

        # protein_pdb_file_name_list, protein_conf_name_list = protein_fixer_runner.run()

        num_protein_models = len(protein_pdb_file_name_list)
        protein_node_list = [None] * num_protein_models
        for protein_model_idx in range(num_protein_models):
            protein_node_dict = {}
            protein_node_dict['protein_pdb_file_name'] = protein_pdb_file_name_list[protein_model_idx]
            protein_node_dict['protein_conf_name'] = protein_conf_name_list[protein_model_idx]
            protein_node_list[protein_model_idx] = protein_node_dict
            # save the mapping file
            raw_protein_name = os.path.basename(self.config_template['protein_pdb_file_name']).replace('.pdb', '')
            fixed_protein_name = os.path.basename(protein_node_dict['protein_conf_name']).replace('.pdb', '')
            mapping_file = 'residues_indices_mapping_{0}_2_{1}.csv'.format(raw_protein_name,
                                                                           fixed_protein_name)
            mapping_file_abs_path = os.path.join(self.config_template['working_dir_name'], mapping_file)
            self._map_residues_indices(self.config_template['protein_pdb_file_name'],
                                       protein_node_dict['protein_pdb_file_name'],
                                       mapping_file_abs_path)

        return protein_node_list

    def _get_chains(self, u):
        segment_ags = u.select_atoms('protein').segments
        seg_res_list = []
        for seg in segment_ags:
            current_seg_id = seg.segid
            res_indices = [res.resid for res in u.select_atoms('segid {0}'.format(current_seg_id)).residues]
            seg_res_list.append(res_indices)
        return seg_res_list

    def _get_mapping_df(self, raw_res_indices, fixed_res_indices,seg_id):
        start_bias = raw_res_indices[0] - fixed_res_indices[0]
        mapping_list = []
        for res_id in fixed_res_indices:
            mapping_list.append([res_id, res_id+start_bias,seg_id])
        return pd.DataFrame(mapping_list, columns=['fixed_res_id','raw_res_id','seg_id'])

    def _map_residues_indices(self, raw_pdb_name, fixed_pdb_name, output_mapping_path):
        """
        :param raw_pdb_name: the path of input pdb file name
        :param fixed_pdb_name:  the path of the pdb file fixed by modeller
        :param output_mapping_path:  the path to save the mapping table(csv)
        :return: None
        """
        if os.path.exists(output_mapping_path):
            print('Warning: the residues indices mapping file {0} is overwritten!'.format(output_mapping_path))
        raw_u = mda.Universe(raw_pdb_name)
        raw_res_list = self._get_chains(raw_u)
        fixed_u = mda.Universe(fixed_pdb_name)
        fixed_res_list = self._get_chains(fixed_u)
        total_list = []
        segment_indices = list(raw_u.atoms.select_atoms('protein').segments.segids)
        for i in range(len(segment_indices)):
            raw_seg = raw_res_list[i]
            fixed_seg = fixed_res_list[i]
            current_seg_id = segment_indices[i]
            total_list.append(self._get_mapping_df(raw_seg, fixed_seg, current_seg_id))
        mapping_table = pd.concat(total_list, axis=0, ignore_index=True)
        mapping_table.to_csv(output_mapping_path, index=False)



class ProteinSystemBuilder(TaskBase):
    def __init__(self, name='protein_system_builder'):
        """Prepare protein hydrogens, resnames, so that the pdb files can be ready for docking, MD, IFP calculations and visualizations.

        Args:
            name (str, optional): the task name. Defaults to 'protein_system_builder'.

        Examples:
        >>> from xdalgorithm.engines import get_dataset
        ... from xdalgorithm.engines import ProteinSystemBuilder
        ... from xdalgorithm.engines import SerialEventBase

        >>> data.run_task(
        ...     SerialEventBase,
        ...     event_type='MODEL',
        ...     task=ProteinSystemBuilder(),
        ...     protein_pdb_file_name='i:protein_pdb_file_name:0.MODEL.protein_fixer',
        ...     protein_conf_name='i:protein_conf_name:0.MODEL.protein_fixer',
        ... )
        """
        super().__init__(name)
        self.config_template = {
            'protein_pdb_file_name': UNDEFINED_PARAMETER,
            'protein_conf_name': UNDEFINED_PARAMETER,
            'kept_ligand_resname_list': None,
            'working_dir_name': None
        }

    def run(self):
        protein_system_builder_runner = ProteinSystemBuilderRunner(protein_pdb_file_name=self.config_template['protein_pdb_file_name'],
                                                                   protein_conf_name=self.config_template['protein_conf_name'],
                                                                   kept_ligand_resname_list=self.config_template['kept_ligand_resname_list'],
                                                                   working_dir_name=self.config_template['working_dir_name'])

        protein_pdb_file_name = protein_system_builder_runner.run()

        return [{'protein_pdb_file_name': protein_pdb_file_name}]

class ScaffoldNetwork(CollectiveTaskBase):
    """Generate a scaffold network and decompose the compounds by their cores.

    Args:
        name (str, optional): the task name. Defaults to 'scaffold_network'.

    Examples:
    >>> from xdalgorithm.engines import get_dataset
    ... from xdalgorithm.engines import ScaffoldNetwork
    ... from xdalgorithm.engines import SerialEventBase, CollectiveEventBase

    >>> data.run_task(
    ...     CollectiveEventBase,
    ...     task=ScaffoldNetwork(),
    ...     event_type='MODEL',
    ...     ligand_molecule_name='i:ligand_molecule_name:0.LIGAND.add_ligands',
    ...     ligand_smiles_string='i:ligand_smiles_string:0.LIGAND.add_ligands'
    ... )
    """
    def __init__(self, name='scaffold_network'):
        super().__init__(name)
        self.config_template = {
            'ligand_molecule_name': UNDEFINED_PARAMETER,
            'ligand_smiles_string': UNDEFINED_PARAMETER,
            'working_dir_name': None
        }

    def collect_config(self, task: TaskBase):
        if isinstance(self.config_template['ligand_molecule_name'], str):
            self.config_template['ligand_molecule_name'] = [self.config_template['ligand_molecule_name'], task.config_template['ligand_molecule_name']]
        elif isinstance(self.config_template['ligand_molecule_name'], list):
            self.config_template['ligand_molecule_name'].append(task.config_template['ligand_molecule_name'])
        else:
            raise TypeError('input ligand_molecule_name expected a list or string.')

        if isinstance(self.config_template['ligand_smiles_string'], str):
            self.config_template['ligand_smiles_string'] = [self.config_template['ligand_smiles_string'], task.config_template['ligand_smiles_string']]
        elif isinstance(self.config_template['ligand_smiles_string'], list):
            self.config_template['ligand_smiles_string'].append(task.config_template['ligand_smiles_string'])
        else:
            raise TypeError('input ligand_smiles_string expected a list or string.')

    def run(self):
        ligand_smiles_string_list = list(self.config_template['ligand_smiles_string'])
        ligand_molecule_name_list = list(self.config_template['ligand_molecule_name'])

        scaffold_network = get_scaffold_network(ligand_smiles_string_list, ligand_molecule_name_list)
        # create a ligand_rgroup_df, indices are the names of compounds
        ligand_rgroup_df = pd.DataFrame(scaffold_network.R_dict_list()[0][0])
        ligand_rgroup_df.index = ligand_molecule_name_list

        node_dict = {}
        node_dict['scaffold_network'] = scaffold_network
        node_dict['ligand_rgroup_df'] = ligand_rgroup_df

        return [node_dict]

class Core(CollectiveTaskBase):
    def __init__(self, name='core'):
        """Add core task, analyse the compounds and find cores for them.

        Args:
            name (str, optional): the task name. Defaults to 'core'.

        Examples:
        >>> from xdalgorithm.engines import get_dataset
        ... from xdalgorithm.engines import Core
        ... from xdalgorithm.engines import SerialEventBase, CollectiveEventBase

        >>> data.run_task(
        ...     CollectiveEventBase,
        ...     task=Core(),
        ...     event_type='MODEL',
        ...     ligand_molecule_name='i:ligand_molecule_name:0.LIGAND.add_ligands',
        ...     ligand_rgroup_df='i:ligand_rgroup_df:0.MODEL.scaffold_network'
        ... )
        """
        super().__init__(name)
        self.config_template = {
            'ligand_molecule_name': UNDEFINED_PARAMETER,
            'ligand_rgroup_df': UNDEFINED_PARAMETER,
            'working_dir_name': None
        }

    def collect_config(self, task: TaskBase):
        if isinstance(self.config_template['ligand_molecule_name'], str):
            self.config_template['ligand_molecule_name'] = [self.config_template['ligand_molecule_name']]
        self.config_template['ligand_molecule_name'].append(task.config_template['ligand_molecule_name'])

    def run(self):
        result_list = []
        for mol_idx, ligand_molecule_name in enumerate(self.config_template['ligand_molecule_name']):
            result_dict = {}
            result_dict['output_ids'] = [mol_idx]
            result_dict['core_smarts_string'] = self.config_template['ligand_rgroup_df'].loc[ligand_molecule_name, 'Core']
            result_list.append(result_dict)

        return result_list

class GenericCore(TaskBase):
    """Get a generic core smarts from scaffold network.

    Args:
        name (str, optional): the task name. Defaults to 'generic_core'.

    Examples:
    >>> from xdalgorithm.engines import get_dataset
    ... from xdalgorithm.engines import GenericCore
    ... from xdalgorithm.engines import SerialEventBase

    >>> data.run_task(
    ...     SerialEventBase,
    ...     task=GenericCore(),
    ...     event_type='MODEL',
    ...     ligand_rgroup_df='i:ligand_rgroup_df:0.MODEL.scaffold_network'
    ... )
    """
    def __init__(self, name='generic_core'):
        super().__init__(name)
        self.config_template = {
            'ligand_rgroup_df': UNDEFINED_PARAMETER,
            'working_dir_name': None
        }

    def run(self):
        reference_ligand_smiles_string = self.config_template['ligand_rgroup_df'].loc[:, 'MOL'].values[0]
        reference_ligand_mol = Chem.MolFromSmiles(reference_ligand_smiles_string)

        core_smarts_string_array = self.config_template['ligand_rgroup_df'].loc[:, 'Core'].values
        core_mol_list = [Chem.MolFromSmiles(core_smarts_string) for core_smarts_string in core_smarts_string_array]

        mcs = rdFMCS.FindMCS(core_mol_list,
                             ringMatchesRingOnly=True,
                             completeRingsOnly=True,
                             atomCompare=rdFMCS.AtomCompare.CompareAnyHeavyAtom,
                             bondCompare=rdFMCS.BondCompare.CompareOrderExact,
                             timeout=60)

        mcs_string = mcs.smartsString.replace('#0', '*')
        generic_core_smarts_string = re.sub('\[\*.*?\]', '[*]', mcs_string)

        generic_core_mol = Chem.MolFromSmarts(generic_core_smarts_string)
        rgroup_decomposition_parameters = rdRGroupDecomposition.RGroupDecompositionParameters()
        rgroup_decomposition_parameters.removeAllHydrogenRGroups = False
        rgroup_decomposition_parameters.removeHydrogensPostMatch = False
        rgroup_decomposition_mol_dict = rdRGroupDecomposition.RGroupDecompose([generic_core_mol],
                                                                              [reference_ligand_mol],
                                                                              asSmiles=False,
                                                                              options=rgroup_decomposition_parameters)[0][0]

        available_rgroup_label_list = list(rgroup_decomposition_mol_dict.keys())

        node_dict = {}
        node_dict['core_smarts_string'] = generic_core_smarts_string
        node_dict['rgroup_label_list'] = available_rgroup_label_list

        return [node_dict]

class RGroupIFPLabels(CollectiveTaskBase):
    def __init__(self, name='rgroup_ifp_labels'):
        """A task to generate rgroup ifp labels as an enumerated numpy unicode array.

        Args:
            name (str, optional): the task name. Defaults to 'rgroup_ifp_labels'.

        Examples:
        >>> from xdalgorithm.engines import get_dataset
        ... from xdalgorithm.engines import RGroupIFPLabels
        ... from xdalgorithm.engines import CollectiveTaskBase

        >>> data.run_task(
        ...     CollectiveTaskBase,
        ...     task=RGroupIFPLabels(),
        ...     protein_pdb_file_name='i:protein_pdb_file_name:0.MODEL.protein_system_builder',
        ...     rgroup_label_list='i:rgroup_label_list:0.MODEL.generic_core'
        ... )
        """
        super().__init__(name)
        self.config_template = {
            'protein_pdb_file_name': UNDEFINED_PARAMETER,
            'rgroup_label_list': UNDEFINED_PARAMETER,
            'ifp_types_list': ['HY', 'IP', 'IN', 'IO', 'AR1', 'AR2', 'AR3', 'HL1', 'HL2', 'HL3', 'HL4', 'RE', 'HD', 'HA'],
            'working_dir_name': None
        }

    def collect_config(self, task: TaskBase):
        if isinstance(self.config_template['protein_pdb_file_name'], str):
            self.config_template['protein_pdb_file_name'] = [self.config_template['protein_pdb_file_name'], task.config_template['protein_pdb_file_name']]
        elif isinstance(self.config_template['protein_pdb_file_name'], list):
            self.config_template['protein_pdb_file_name'].append(task.config_template['protein_pdb_file_name'])
        else:
            raise TypeError('input protein_pdb_file_name expected a list or string.')

    def run(self):
        num_source_nodes = len(self.config_template['protein_pdb_file_name'])
        protein_ag = mda.Universe(self.config_template['protein_pdb_file_name'][0]).atoms
        protein_resids_array = protein_ag.residues.resids
        protein_resnames_array = protein_ag.residues.resnames.astype('U')
        num_protein_residues = protein_ag.residues.n_residues

        rgroup_ifp_label_list = []
        for ifp_type in self.config_template['ifp_types_list']:
            for protein_residue_idx in range(num_protein_residues):
                protein_resname = protein_resnames_array[protein_residue_idx]
                protein_resid = protein_resids_array[protein_residue_idx]
                for rgroup_label in self.config_template['rgroup_label_list']:
                    current_rgroup_ifp_label = ifp_type + ':' + protein_resname + str(protein_resid) + '...' + rgroup_label
                    rgroup_ifp_label_list.append(current_rgroup_ifp_label)

        rgroup_ifp_label_array = np.array(rgroup_ifp_label_list, dtype='U')

        node_dict = {}
        node_dict['output_ids'] = list(range(num_source_nodes))
        node_dict['rgroup_ifp_label_array'] = rgroup_ifp_label_array

        return [node_dict]
