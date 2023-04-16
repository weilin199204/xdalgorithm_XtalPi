import sys
import os
import numpy as np
import pandas as pd

from rdkit import Chem

class InterMoleculeRMSDClustering(object):
    def __init__(self, parameters_dict):
        protein_ligand_conf_csv_file_name = parameters_dict.get('protein_ligand_conf_csv', ValueError)
        self.protein_ligand_conf_df = pd.read_csv(protein_ligand_conf_csv_file_name)
        self.core_smarts_string = parameters_dict.get('core_smarts_string', ValueError)
        self.cluster_method = parameters_dict.get('cluster_method', 'complete')
        self.cluster_cutoff = parameters_dict.get('cluster_cutoff', 2.0)
        self.cluster_criterion = parameters_dict.get('cluster_criterion', 'distance')
        self.working_dir = os.path.abspath(parameters_dict.get('working_dir', '.'))

    def run(self):
        from xdalgorithm.toolbox.clustering.inter_molecule_rmsd_clustering_runner import InterMoleculeRMSDClusteringRunner

        protein_pdb_file_name_array = self.protein_ligand_conf_df.loc[:, 'protein_pdb_file_name'].values.astype('U')
        unique_protein_pdb_file_name_array = np.unique(protein_pdb_file_name_array)
        num_protein_conformations = unique_protein_pdb_file_name_array.shape[0]
        cluster_representatives_df_list = []

        for protein_conf_idx in range(num_protein_conformations):
            current_protein_conf_working_dir_name = os.path.join(self.working_dir, 'protein_conf_' + str(protein_conf_idx))
            os.mkdir(current_protein_conf_working_dir_name)
            current_protein_pdb_file_name = unique_protein_pdb_file_name_array[protein_conf_idx]
            current_protein_conf_name = 'protein_conf_' + str(protein_conf_idx)
            current_protein_conf_data_df = self.protein_ligand_conf_df.iloc[(self.protein_ligand_conf_df.loc[:, 'protein_pdb_file_name'] == current_protein_pdb_file_name).values, :]
            current_ligand_sdf_file_name_list = current_protein_conf_data_df.loc[:, 'ligand_sdf_file_name'].values.tolist()

            current_inter_molecule_rmsd_clustering_runner = InterMoleculeRMSDClusteringRunner(current_ligand_sdf_file_name_list,
                                                                                      self.core_smarts_string,
                                                                                      cluster_method=self.cluster_method,
                                                                                      cluster_cutoff=self.cluster_cutoff,
                                                                                      cluster_criterion=self.cluster_criterion,
                                                                                      working_dir_name=current_protein_conf_working_dir_name)

            _, current_cluster_info_df = current_inter_molecule_rmsd_clustering_runner.run()
            current_num_clusters = current_cluster_info_df.shape[0]
            current_cluster_info_df['protein_pdb_file_name'] = [current_protein_pdb_file_name] * current_num_clusters
            current_cluster_info_df.drop(['cluster_representative_conf_idx', 'cluster_member_sdf_file_idx'], axis=1, inplace=True)
            current_cluster_info_df.rename(columns={'cluster_representative_sdf_file_name': 'ligand_sdf_file_name'}, inplace=True)
            cluster_representatives_df_list.append(current_cluster_info_df)

        cluster_representatives_df = pd.concat(cluster_representatives_df_list)
        cluster_representatives_df.reset_index(drop=True, inplace=True)
        cluster_representatives_csv_file_name = os.path.join(self.working_dir, 'cluster_representatives.csv')
        cluster_representatives_df.to_csv(cluster_representatives_csv_file_name, index=False)

class IntraMoleculeRMSDClustering(object):
    def __init__(self, parameters_dict):
        protein_ligand_conf_csv_file_name = parameters_dict.get('protein_ligand_conf_csv', ValueError)
        self.protein_ligand_conf_df = pd.read_csv(protein_ligand_conf_csv_file_name)
        self.cluster_method = parameters_dict.get('cluster_method', 'complete')
        self.cluster_cutoff = parameters_dict.get('cluster_cutoff', 2.0)
        self.cluster_criterion = parameters_dict.get('cluster_criterion', 'distance')
        self.working_dir = os.path.abspath(parameters_dict.get('working_dir', '.'))

    def __group_conformations_by_smiles_strings__(self, ligand_smiles_string_list):
        if ligand_smiles_string_list ==  [None] * len(ligand_smiles_string_list):
            num_molecules = len(ligand_smiles_string_list)
            ligand_conf_idx_group_list = [None] * num_molecules

            for mol_idx in range(num_molecules):
                ligand_conf_idx_group_list[mol_idx] = [mol_idx]

        else:
            ligand_smiles_string_array = np.array(ligand_smiles_string_list, dtype='U')
            ligand_unique_smiles_string_array, ligand_unique_mapping_array = np.unique(ligand_smiles_string_array, return_inverse=True)

            num_molecules = ligand_unique_smiles_string_array.shape[0]
            ligand_conf_idx_group_list = [None] * num_molecules

            for mol_idx in range(num_molecules):
                current_mapping_idx_list = np.where(ligand_unique_mapping_array == mol_idx)[0].tolist()
                ligand_conf_idx_group_list[mol_idx] = current_mapping_idx_list

        return ligand_conf_idx_group_list

    def run(self):
        from xdalgorithm.toolbox.clustering.intra_molecule_rmsd_clustering_runner import IntraMoleculeRMSDClusteringRunner

        protein_pdb_file_name_array = self.protein_ligand_conf_df.loc[:, 'protein_pdb_file_name'].values.astype('U')
        unique_protein_pdb_file_name_array = np.unique(protein_pdb_file_name_array)
        num_protein_conformations = unique_protein_pdb_file_name_array.shape[0]
        cluster_representatives_df_list = []

        for protein_conf_idx in range(num_protein_conformations):
            current_protein_conf_working_dir_name = os.path.join(self.working_dir, 'protein_conf_' + str(protein_conf_idx))
            os.mkdir(current_protein_conf_working_dir_name)
            current_protein_pdb_file_name = unique_protein_pdb_file_name_array[protein_conf_idx]
            current_protein_conf_name = 'protein_conf_' + str(protein_conf_idx)
            current_protein_conf_data_df = self.protein_ligand_conf_df.iloc[(self.protein_ligand_conf_df.loc[:, 'protein_pdb_file_name'] == current_protein_pdb_file_name).values, :]
            current_ligand_sdf_file_name_list = current_protein_conf_data_df.loc[:, 'ligand_sdf_file_name'].values.tolist()
            current_ligand_sdf_file_name_array = np.array(current_ligand_sdf_file_name_list, dtype='U')

            num_sdf_files = len(current_ligand_sdf_file_name_list)
            current_ligand_smiles_string_list = [None] * num_sdf_files
            for sdf_file_idx in range(num_sdf_files):
                ligand_sdf_file_name = current_ligand_sdf_file_name_list[sdf_file_idx]
                mol = Chem.SDMolSupplier(ligand_sdf_file_name, removeHs=False)[0]
                current_ligand_smiles_string_list[sdf_file_idx] = Chem.MolToSmiles(mol, allHsExplicit=True)

            current_ligand_conf_idx_group_list = self.__group_conformations_by_smiles_strings__(current_ligand_smiles_string_list)
            current_num_molecules = len(current_ligand_conf_idx_group_list)
            current_ligand_sdf_file_name_group_list = [None] * current_num_molecules

            for mol_idx in range(current_num_molecules):
                ligand_conf_idx_list = current_ligand_conf_idx_group_list[mol_idx]
                ligand_sdf_file_name_list = current_ligand_sdf_file_name_array[ligand_conf_idx_list].tolist()
                ligand_molecule_working_dir_name = os.path.join(current_protein_conf_working_dir_name, 'ligand_isomer_' + str(mol_idx))
                os.mkdir(ligand_molecule_working_dir_name)

                intra_molecule_rmsd_clustering_runner = IntraMoleculeRMSDClusteringRunner(ligand_sdf_file_name_list,
                                                                                          cluster_method=self.cluster_method,
                                                                                          cluster_cutoff=self.cluster_cutoff,
                                                                                          cluster_criterion=self.cluster_criterion,
                                                                                          working_dir_name=ligand_molecule_working_dir_name)

                _, cluster_info_df = intra_molecule_rmsd_clustering_runner.run()
                num_clusters = cluster_info_df.shape[0]
                cluster_info_df['protein_pdb_file_name'] = [current_protein_pdb_file_name] * num_clusters
                cluster_info_df.drop(['cluster_representative_conf_idx', 'cluster_member_sdf_file_idx'], axis=1, inplace=True)
                cluster_info_df.rename(columns={'cluster_representative_sdf_file_name': 'ligand_sdf_file_name'}, inplace=True)
                cluster_representatives_df_list.append(cluster_info_df)

        cluster_representatives_df = pd.concat(cluster_representatives_df_list)
        cluster_representatives_df.reset_index(drop=True, inplace=True)
        cluster_representatives_csv_file_name = os.path.join(self.working_dir, 'cluster_representatives.csv')
        cluster_representatives_df.to_csv(cluster_representatives_csv_file_name, index=False)

class InterMoleculeIFPClustering(object):
    def __init__(self, parameters_dict):
        protein_ligand_conf_csv_file_name = parameters_dict.get('protein_ligand_conf_csv', ValueError)
        self.protein_ligand_conf_df = pd.read_csv(protein_ligand_conf_csv_file_name)
        self.ligand_resname = parameters_dict.get('ligand_resname', 'MOL')
        self.include_general_contacts = parameters_dict.get('include_general_contacts', False)

        if parameters_dict['ifp_clustering_type'] == 'rgroup':
            self.ifp_clustering_type = 'rgroup'
            self.core_smarts_string = parameters_dict.get('core_smarts_string', ValueError)
        elif parameters_dict['ifp_clustering_type'] == 'whole_molecule':
            self.ifp_clustering_type = 'whole_molecule'
        else:
            raise ValueError('Specified ifp_clustering_type NOT supported.')

        self.metric = parameters_dict.get('metric', 'jaccard')
        self.cluster_method = parameters_dict.get('cluster_method', 'ward')
        self.cluster_cutoff = parameters_dict.get('cluster_cutoff', None)
        self.cluster_criterion = parameters_dict.get('cluster_criterion', 'distance')
        self.working_dir = os.path.abspath(parameters_dict.get('working_dir', '.'))

    def run(self):
        from xdalgorithm.toolbox.interaction_fingerprints.ifp_calculations_runner import IFPCalculationsRunner
        from xdalgorithm.toolbox.interaction_fingerprints.rgroup_ifp_analysis_runner import RGroupIFPAnalysisRunner
        from xdalgorithm.toolbox.interaction_fingerprints.whole_molecule_ifp_analysis_runner import WholeMoleculeIFPAnalysisRunner
        from xdalgorithm.toolbox.clustering.inter_molecule_ifp_clustering_runner import InterMoleculeIFPClusteringRunner

        protein_pdb_file_name_array = self.protein_ligand_conf_df.loc[:, 'protein_pdb_file_name'].values.astype('U')
        unique_protein_pdb_file_name_array = np.unique(protein_pdb_file_name_array)
        num_protein_conformations = unique_protein_pdb_file_name_array.shape[0]

        complex_ligand_smiles_string_dict = {}
        complex_ligand_sdf_file_name_dict = {}
        for protein_conf_idx in range(num_protein_conformations):
            current_protein_pdb_file_name = unique_protein_pdb_file_name_array[protein_conf_idx]
            current_protein_conf_name = 'protein_conf_' + str(protein_conf_idx)
            current_protein_conf_data_df = self.protein_ligand_conf_df.iloc[(self.protein_ligand_conf_df.loc[:, 'protein_pdb_file_name'] == current_protein_pdb_file_name).values, :]
            current_ligand_sdf_file_name_list = current_protein_conf_data_df.loc[:, 'ligand_sdf_file_name'].values.tolist()

            current_protein_conf_fixed_sdf_dir_name = os.path.join(self.working_dir, current_protein_conf_name + '_fixed_sdf')
            os.mkdir(current_protein_conf_fixed_sdf_dir_name)
            num_current_sdf_files = len(current_ligand_sdf_file_name_list)
            current_fixed_ligand_smiles_string_list = []
            current_fixed_ligand_sdf_file_name_list = []
            for sdf_file_idx in range(num_current_sdf_files):
                ligand_sdf_file_name = current_ligand_sdf_file_name_list[sdf_file_idx]
                ligand_sdf_file_prefix = os.path.basename(ligand_sdf_file_name).split('.')[0]
                mol_list = Chem.SDMolSupplier(ligand_sdf_file_name, removeHs=False)
                if len(mol_list) > 1:
                    for idx, mol in enumerate(mol_list):
                        fixed_ligand_smiles_string = Chem.MolToSmiles(mol, allHsExplicit=True)
                        fixed_ligand_sdf_file_name = os.path.join(current_protein_conf_fixed_sdf_dir_name, ligand_sdf_file_prefix + '_' + str(idx))
                        current_fixed_ligand_smiles_string_list.append(fixed_ligand_smiles_string)
                        current_fixed_ligand_sdf_file_name_list.append(fixed_ligand_sdf_file_name)
                        sdf_writer = Chem.SDWriter(fixed_ligand_sdf_file_name)
                        sdf_writer.write(mol)
                        sdf_writer.flush()
                        sdf_writer.close()
                else:
                    current_fixed_ligand_smiles_string_list.append(Chem.MolToSmiles(mol_list[0], allHsExplicit=True))
                    current_fixed_ligand_sdf_file_name_list.append(ligand_sdf_file_name)

            complex_ligand_smiles_string_dict[current_protein_pdb_file_name] = current_fixed_ligand_smiles_string_list
            complex_ligand_sdf_file_name_dict[current_protein_pdb_file_name] = current_fixed_ligand_sdf_file_name_list

        sys.stdout = open(os.path.join(self.working_dir, 'ifp_clustering.log'), 'w')
        sys.stderr = open(os.path.join(self.working_dir, 'ifp_clustering_error.log'), 'w')
        ifp_calculations_runner = IFPCalculationsRunner(complex_ligand_sdf_file_name_dict=complex_ligand_sdf_file_name_dict,
                                                        complex_ligand_smiles_string_dict=complex_ligand_smiles_string_dict,
                                                        ligand_resname=self.ligand_resname,
                                                        ligand_charge_method='gas',
                                                        include_general_contacts=self.include_general_contacts,
                                                        donors_selection_string='name N* or name O*',
                                                        hydrogens_selection_string='name H*',
                                                        acceptors_selection_string='name F* or name N* or name O*',
                                                        donor_hydrogen_distance_cutoff=2.0,
                                                        donor_acceptor_distance_cutoff=3.6,
                                                        donor_hydrogen_acceptor_angle_cutoff=120.0,
                                                        working_dir_name=self.working_dir)

        complex_ifp_analysis_summary_df_dict, complex_ligand_sdf_file_name_group_dict, complex_ligand_smiles_string_group_dict = ifp_calculations_runner.run()

        flattened_complex_ifp_string_nested_list_dict = {}
        flattened_complex_ligand_smiles_string_list_dict = {}
        flattened_complex_ligand_sdf_file_name_list_dict = {}

        for protein_conf_idx in range(num_protein_conformations):
            current_protein_pdb_file_name = unique_protein_pdb_file_name_array[protein_conf_idx]
            current_ifp_dir_name = 'protein_conf_' + str(protein_conf_idx) + '_ifp'
            current_working_dir_name = os.path.join(self.working_dir, current_ifp_dir_name)
            os.mkdir(current_working_dir_name)

            current_ifp_analysis_summary_df_list = complex_ifp_analysis_summary_df_dict[current_protein_pdb_file_name]
            current_ligand_sdf_file_name_group_list = complex_ligand_sdf_file_name_group_dict[current_protein_pdb_file_name]
            current_ligand_smiles_string_group_list = complex_ligand_smiles_string_group_dict[current_protein_pdb_file_name]

            flattened_complex_ligand_smiles_string_list_dict[current_protein_pdb_file_name] = []
            flattened_complex_ligand_sdf_file_name_list_dict[current_protein_pdb_file_name] = []
            flattened_complex_ifp_string_nested_list_dict[current_protein_pdb_file_name] = []

            num_molecules = len(current_ligand_sdf_file_name_group_list)
            for mol_idx in range(num_molecules):
                molecule_name = os.path.basename(current_ligand_sdf_file_name_group_list[mol_idx][0]).split('.')[0]
                ifp_file_name = os.path.join(current_working_dir_name, molecule_name + '_ifp.dat')

                ligand_sdf_file_name_list = current_ligand_sdf_file_name_group_list[mol_idx]
                ligand_smiles_string_list = current_ligand_smiles_string_group_list[mol_idx]
                flattened_complex_ligand_smiles_string_list_dict[current_protein_pdb_file_name].extend(ligand_smiles_string_list)
                flattened_complex_ligand_sdf_file_name_list_dict[current_protein_pdb_file_name].extend(ligand_sdf_file_name_list)

                current_ifp_analysis_summary_df = current_ifp_analysis_summary_df_list[mol_idx]
                ifp_description_string_nested_list = current_ifp_analysis_summary_df.loc[:, 'ifp_description_string_list'].values.tolist()
                flattened_complex_ifp_string_nested_list_dict[current_protein_pdb_file_name].extend(ifp_description_string_nested_list)

                num_conformations = len(ligand_sdf_file_name_list)
                for conf_idx in range(num_conformations):
                    ifp_description_string_list = ifp_description_string_nested_list[conf_idx]
                    ligand_sdf_file_name = ligand_sdf_file_name_list[conf_idx]
                    molecule_name = os.path.basename(ligand_sdf_file_name).split('.')[0]
                    ifp_file_name = os.path.join(current_working_dir_name, molecule_name + '_ifp.dat')
                    np.savetxt(ifp_file_name, ifp_description_string_list, fmt='%s')

        cluster_representatives_df_list = []
        for protein_conf_idx in range(num_protein_conformations):
            current_clustering_path_prefix = os.path.join(self.working_dir, 'protein_conf_' + str(protein_conf_idx) + '_clustering')
            os.mkdir(current_clustering_path_prefix)
            current_protein_pdb_file_name = unique_protein_pdb_file_name_array[protein_conf_idx]
            current_ligand_smiles_string_list = flattened_complex_ligand_smiles_string_list_dict[current_protein_pdb_file_name]
            current_ligand_sdf_file_name_list = flattened_complex_ligand_sdf_file_name_list_dict[current_protein_pdb_file_name]
            current_ifp_description_string_nested_list = flattened_complex_ifp_string_nested_list_dict[current_protein_pdb_file_name]

            current_ligand_sdf_file_name_array = np.array(current_ligand_sdf_file_name_list, dtype='U')

            if self.ifp_clustering_type == 'rgroup':
                rgroup_ifp_analysis_runner = RGroupIFPAnalysisRunner(ligand_smiles_string_list=current_ligand_smiles_string_list,
                                                                     ifp_description_string_nested_list=current_ifp_description_string_nested_list,
                                                                     core_smarts_string=self.core_smarts_string)

                rgroup_ifp_vector_group_list, _, _, _ = rgroup_ifp_analysis_runner.run()
                flattened_ifp_vector_list = [rgroup_ifp_vector for rgroup_ifp_vector_list in rgroup_ifp_vector_group_list for rgroup_ifp_vector in rgroup_ifp_vector_list]
            elif self.ifp_clustering_type == 'whole_molecule':
                whole_molecule_ifp_analysis_runner = WholeMoleculeIFPAnalysisRunner(current_ifp_description_string_nested_list)
                flattened_ifp_vector_list, _ = whole_molecule_ifp_analysis_runner.run()

            inter_molecule_ifp_clustering_runner = InterMoleculeIFPClusteringRunner(flattened_ifp_vector_list,
                                                                                    metric=self.metric,
                                                                                    cluster_method=self.cluster_method,
                                                                                    cluster_cutoff=self.cluster_cutoff,
                                                                                    cluster_criterion=self.cluster_criterion,
                                                                                    working_dir_name=current_clustering_path_prefix)

            cluster_info_df = inter_molecule_ifp_clustering_runner.run()

            num_clusters = cluster_info_df.shape[0]
            cluster_representative_conf_idx_list = cluster_info_df.loc[:, 'cluster_representative_conf_idx'].values.tolist()
            cluster_representative_sdf_file_name_list = current_ligand_sdf_file_name_array[cluster_representative_conf_idx_list].tolist()
            cluster_info_df['protein_pdb_file_name'] = [current_protein_pdb_file_name] * num_clusters
            cluster_info_df['ligand_sdf_file_name'] = cluster_representative_sdf_file_name_list
            cluster_info_df.drop(['cluster_representative_conf_idx', 'cluster_member_ifp_idx'], axis=1, inplace=True)
            cluster_representatives_df_list.append(cluster_info_df)

        cluster_representatives_df = pd.concat(cluster_representatives_df_list)
        cluster_representatives_df.reset_index(drop=True, inplace=True)
        cluster_representatives_csv_file_name = os.path.join(self.working_dir, 'cluster_representatives.csv')
        cluster_representatives_df.to_csv(cluster_representatives_csv_file_name, index=False)

        sys.stdout.close()
        sys.stderr.close()

class CLICommand:
    """Cluster the conformations according to the RMSD or IFP distance.

    The conformers in a list of sdf files are loaded. The hierarchical clustering, which is implemented by
    scipy libraray in python, clusters the conformers according the the RMSD or IFP distance.

    **Features:**
    Three modes are provided:
    1) 'inter_molecule_rmsd_clustering'
    2) 'intra_molecule_rmsd_clustering'
    3) 'inter_molecule_ifp_clustering'

    -----------------------------------------------------------------------------------------------
    Input json template for intra_molecule_rmsd_clustering:
    {"run_type": "intra_molecule_rmsd_clustering",
     "parameters":
        {
         "protein_ligand_conf_csv": "protein_ligand_conf.csv",
         "cluster_method": "complete",
         "cluster_cutoff": 2.0,
         "cluster_criterion": "distance",
         "working_dir": "."
        }
    }
    ------------------------------------------------------------------------------------------------
    Input json template for inter_molecule_rmsd_clustering:
    {"run_type": "inter_molecule_rmsd_clustering",
     "parameters":
        {
         "protein_ligand_conf_csv": "protein_ligand_conf.csv",
         "core_smarts_string": "[#6&R]1:&@[#6&R]:&@[#7&R]:&@[#6&R](:&@[#6&R](:&@[#6&R]:&@1)-&!@[#6&R]1:&@[#7&R]:&@[#6&R]2:&@[#6&R](:&@[#7&R]:&@1-&!@[#6&R]1:&@[#6&R]:&@[#6&R]:&@[#6&R]:&@[#6&R]:&@[#6&R]:&@1):&@[#7,#6;R]:&@[#6&R]:&@[#6,#7;R]:&@[#6&R]:&@2)-&!@[*&!R]",
         "cluster_method": "complete",
         "cluster_cutoff": 2.0,
         "cluster_criterion": "distance",
         "working_dir": "."
        }
    }
    ------------------------------------------------------------------------------------------------
    Input json template for inter_molecule_ifp_clustering:
    {"run_type": "inter_molecule_ifp_clustering",
     "parameters":
        {
         "protein_ligand_conf_csv": "protein_ligand_conf.csv",
         "ligand_resname": "MOL",
         "include_general_contacts": false,
         "ifp_clustering_type": "rgroup",
         "core_smarts_string": "[#6&R]1:&@[#6&R]:&@[#7&R]:&@[#6&R](:&@[#6&R](:&@[#6&R]:&@1)-&!@[#6&R]1:&@[#7&R]:&@[#6&R]2:&@[#6&R](:&@[#7&R]:&@1-&!@[#6&R]1:&@[#6&R]:&@[#6&R]:&@[#6&R]:&@[#6&R]:&@[#6&R]:&@1):&@[#7,#6;R]:&@[#6&R]:&@[#6,#7;R]:&@[#6&R]:&@2)-&!@[*&!R]",
         "metric": "jaccard",
         "cluster_method": "ward",
         "cluster_cutoff": null,
         "cluster_criterion": "distance",
         "working_dir": "."
        }
    }
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-f', '--input-json', default='', type=str)

    @staticmethod
    def run(args):
        from xdalgorithm.utils import load_arguments_from_json, save_arguments_to_json
        configuration = load_arguments_from_json(args.input_json)
        run_type = configuration['run_type']
        input_parameters = configuration['parameters']

        runner_dict = {
            'inter_molecule_rmsd_clustering': InterMoleculeRMSDClustering,
            'intra_molecule_rmsd_clustering': IntraMoleculeRMSDClustering,
            'inter_molecule_ifp_clustering': InterMoleculeIFPClustering
        }

        runner = runner_dict[run_type](input_parameters)
        runner.run()
