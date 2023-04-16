import sys
import os
import numpy as np
import pandas as pd

from rdkit import Chem

class CLICommand:
    """Perform interaction analysis calculations for given protein-ligand docking poses.

    Input json template:
    {
        "parameters":
        {
            "protein_ligand_conf_csv ": "XXXX.csv",
            "ligand_resname": "MOL",
            "include_general_contacts": false,
            "working_dir": "."
        }
    }
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-f', '--input-json', default='', type=str)

    @staticmethod
    def run(args):
        from xdalgorithm.utils import load_arguments_from_json
        from xdalgorithm.toolbox.interaction_fingerprints.ifp_calculations_runner import IFPCalculationsRunner
        configuration = load_arguments_from_json(args.input_json)

        input_parameters = configuration['parameters']
        protein_ligand_conf_df = pd.read_csv(input_parameters['protein_ligand_conf_csv'])
        protein_pdb_file_name_array = protein_ligand_conf_df.loc[:, 'protein_pdb_file_name'].values.astype('U')
        unique_protein_pdb_file_name_array = np.unique(protein_pdb_file_name_array)
        num_protein_conformations = unique_protein_pdb_file_name_array.shape[0]

        complex_ligand_smiles_string_dict = {}
        complex_ligand_sdf_file_name_dict = {}
        for protein_conf_idx in range(num_protein_conformations):
            current_protein_pdb_file_name = unique_protein_pdb_file_name_array[protein_conf_idx]
            current_protein_conf_name = 'protein_conf_' + str(protein_conf_idx)
            current_protein_conf_data_df = protein_ligand_conf_df.iloc[(protein_ligand_conf_df.loc[:, 'protein_pdb_file_name'] == current_protein_pdb_file_name).values, :]
            current_ligand_sdf_file_name_list = current_protein_conf_data_df.loc[:, 'ligand_sdf_file_name'].values.tolist()

            current_protein_conf_fixed_sdf_dir_name = os.path.join(input_parameters['working_dir'], current_protein_conf_name + '_fixed_sdf')
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

        sys.stdout = open(os.path.join(input_parameters['working_dir'], 'ifp.log'), 'w')
        sys.stderr = open(os.path.join(input_parameters['working_dir'], 'ifp_error.log'), 'w')
        ifp_calculations_runner = IFPCalculationsRunner(complex_ligand_sdf_file_name_dict=complex_ligand_sdf_file_name_dict,
                                                        complex_ligand_smiles_string_dict=complex_ligand_smiles_string_dict,
                                                        ligand_resname=input_parameters['ligand_resname'],
                                                        ligand_charge_method='gas',
                                                        include_general_contacts=input_parameters['include_general_contacts'],
                                                        donors_selection_string='name N* or name O*',
                                                        hydrogens_selection_string='name H*',
                                                        acceptors_selection_string='name F* or name N* or name O*',
                                                        donor_hydrogen_distance_cutoff=2.0,
                                                        donor_acceptor_distance_cutoff=3.6,
                                                        donor_hydrogen_acceptor_angle_cutoff=120.0,
                                                        working_dir_name=input_parameters['working_dir'])

        complex_ifp_analysis_summary_df_dict, complex_ligand_sdf_file_name_group_dict, complex_ligand_smiles_string_group_dict = ifp_calculations_runner.run()

        flattened_complex_ifp_string_nested_list_dict = {}
        flattened_complex_ligand_smiles_string_list_dict = {}

        for protein_conf_idx in range(num_protein_conformations):
            current_protein_pdb_file_name = unique_protein_pdb_file_name_array[protein_conf_idx]
            current_ifp_dir_name = 'protein_conf_' + str(protein_conf_idx) + '_ifp'
            current_working_dir_name = os.path.join(input_parameters['working_dir'], current_ifp_dir_name)
            os.mkdir(current_working_dir_name)

            current_ifp_analysis_summary_df_list = complex_ifp_analysis_summary_df_dict[current_protein_pdb_file_name]
            current_ligand_sdf_file_name_group_list = complex_ligand_sdf_file_name_group_dict[current_protein_pdb_file_name]
            current_ligand_smiles_string_group_list = complex_ligand_smiles_string_group_dict[current_protein_pdb_file_name]

            flattened_complex_ligand_smiles_string_list_dict[current_protein_pdb_file_name] = []
            flattened_complex_ifp_string_nested_list_dict[current_protein_pdb_file_name] = []

            num_molecules = len(current_ligand_sdf_file_name_group_list)
            for mol_idx in range(num_molecules):
                molecule_name = os.path.basename(current_ligand_sdf_file_name_group_list[mol_idx][0]).split('.')[0]
                ifp_file_name = os.path.join(current_working_dir_name, molecule_name + '_ifp.dat')

                ligand_sdf_file_name_list = current_ligand_sdf_file_name_group_list[mol_idx]
                ligand_smiles_string_list = current_ligand_smiles_string_group_list[mol_idx]
                flattened_complex_ligand_smiles_string_list_dict[current_protein_pdb_file_name].extend(ligand_smiles_string_list)

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

        sys.stdout.close()
        sys.stderr.close()
