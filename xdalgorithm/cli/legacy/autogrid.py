import os
import pandas as pd

class CLICommand:
    """ Generate docking grids for certain protein conformations given by input pdb file

    Input json template:
    {
        "parameters":
        {
            "protein_conf_csv": "XXXX.csv",
            "kept_ligand_resname_list": null,
            "target_center": [0.0, 0.0, 0.0],
            "num_grid_points": [60, 60, 60],
            "grid_spacing": [0.375, 0.375, 0.375],
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
        from xdalgorithm.toolbox.docking_engines.autogrid_runner import AutoGridRunner
        configuration = load_arguments_from_json(args.input_json)

        input_parameters = configuration['parameters']
        input_protein_df = pd.read_csv(input_parameters['protein_conf_csv'])
        protein_conf_name_list = input_protein_df.loc[:, 'protein_conf_name'].values.tolist()
        protein_pdb_file_name_list = input_protein_df.loc[:, 'protein_pdb_file_name'].values.tolist()
        num_protein_conformations = len(protein_conf_name_list)
        protein_grid_file_name_list = [None] * num_protein_conformations

        for protein_conf_idx in range(num_protein_conformations):
            current_protein_conf_name = protein_conf_name_list[protein_conf_idx]
            current_protein_pdb_file_name = protein_pdb_file_name_list[protein_conf_idx]
            current_autogrid_runner = AutoGridRunner(protein_pdb_file_name=current_protein_pdb_file_name,
                                                     protein_conf_name=current_protein_conf_name,
                                                     kept_ligand_resname_list=input_parameters['kept_ligand_resname_list'],
                                                     target_center=tuple(input_parameters['target_center']),
                                                     num_grid_points=tuple(input_parameters['num_grid_points']),
                                                     grid_spacing=tuple(input_parameters['grid_spacing']),
                                                     working_dir_name=input_parameters['working_dir'])

            protein_grid_file_name_list[protein_conf_idx] = current_autogrid_runner.run()

        input_protein_df['protein_grid_maps_fld_file_name'] = protein_grid_file_name_list

        returned_protein_info_csv_file_name = os.path.join(input_parameters['working_dir'], 'protein_conf_grids.csv')
        input_protein_df.to_csv(returned_protein_info_csv_file_name, index=False)
