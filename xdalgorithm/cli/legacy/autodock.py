import os
import pandas as pd

class CLICommand:
    """Perform docking calculations for given protein-ligand conformation pairs.

    Input json template:
    {
        "parameters":
        {
            "protein_conf_csv ": "XXXX.csv",
            "ligand_conf_csv ": "XXXX.csv",
            "num_docking_runs ": 10,
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
        from xdalgorithm.toolbox.docking_engines.autodock_runner import AutoDockRunner
        configuration = load_arguments_from_json(args.input_json)

        input_parameters = configuration['parameters']
        input_protein_df = pd.read_csv(input_parameters['protein_conf_csv'])
        input_ligand_df = pd.read_csv(input_parameters['ligand_conf_csv'])
        protein_pdb_name_list = input_protein_df.loc[:, 'protein_pdb_file_name'].values.tolist()
        protein_grid_maps_fld_file_name_list = input_protein_df.loc[:, 'protein_grid_maps_fld_file_name'].values.tolist()
        ligand_sdf_file_name_list = input_ligand_df.loc[:, 'ligand_sdf_file_name'].values.tolist()
        num_protein_conformations = len(protein_grid_maps_fld_file_name_list)

        autodock_runner = AutoDockRunner(protein_grid_maps_fld_file_name_list=protein_grid_maps_fld_file_name_list,
                                         ligand_sdf_file_name_list=ligand_sdf_file_name_list,
                                         num_docking_runs=input_parameters['num_docking_runs'],
                                         working_dir_name=input_parameters['working_dir'])

        docking_pose_summary_info_df_list = autodock_runner.run()
        complex_summary_info_df_list = [None] * num_protein_conformations

        for protein_conf_idx in range(num_protein_conformations):
            current_protein_pdb_file_name = protein_pdb_name_list[protein_conf_idx]
            current_protein_grid_file_name = protein_grid_maps_fld_file_name_list[protein_conf_idx]
            current_docking_pose_summary_info_df = docking_pose_summary_info_df_list[protein_conf_idx]
            num_ligand_docking_poses = current_docking_pose_summary_info_df.shape[0]
            current_docking_pose_summary_info_df['protein_pdb_file_name'] = [current_protein_pdb_file_name] * num_ligand_docking_poses
            current_docking_pose_summary_info_df['protein_grid_maps_fld_file_name'] = [current_protein_grid_file_name] * num_ligand_docking_poses
            complex_summary_info_df_list[protein_conf_idx] = current_docking_pose_summary_info_df

        merged_complex_summary_info_df = pd.concat(complex_summary_info_df_list)
        merged_complex_summary_info_df.reset_index(drop=True, inplace=True)
        merged_complex_summary_info_csv_file_name = os.path.join(input_parameters['working_dir'], 'docking_pose_summary.csv')
        merged_complex_summary_info_df.to_csv(merged_complex_summary_info_csv_file_name, index=False)
