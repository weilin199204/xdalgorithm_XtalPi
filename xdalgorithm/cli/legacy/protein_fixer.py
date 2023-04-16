import os
import pandas as pd

class CLICommand:
    """Fix protein structure, typical processes involve removing undesired chains, removing heteroatoms, filling missing residues, etc.

    Input json template:
    {
        "parameters":
        {
            "protein_pdb_file_name": "XXXX.pdb",
            "alignment_option": "all",
            "keep_water": "none",
            "fill_ter_residue": false,
            "cap_residue": false,
            "long_loop": 5,
            "patch_ter": false,
            "num_models": 1,
            "removed_chain_id_list": null,
            "kept_ligand_resname_list": null,
            "reference_pdb_file_name_for_alignment": null,
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
        from xdalgorithm.toolbox.protein_fixer.protein_fixer_runner import ProteinFixerRunner
        from xdalgorithm.toolbox.md.system_building.protein_system_builder_runner import ProteinSystemBuilderRunner
        configuration = load_arguments_from_json(args.input_json)

        input_parameters = configuration['parameters']

        protein_fixer = ProteinFixerRunner(protein_pdb_file_name=input_parameters['protein_pdb_file_name'],
                                           alignment_option=input_parameters['alignment_option'],
                                           keep_water=input_parameters['keep_water'],
                                           fill_ter_residue=input_parameters['fill_ter_residue'],
                                           cap_residue=input_parameters['cap_residue'],
                                           long_loop=input_parameters['long_loop'],
                                           patch_ter=input_parameters['patch_ter'],
                                           num_models=input_parameters['num_models'],
                                           removed_chain_id_list=input_parameters['removed_chain_id_list'],
                                           kept_ligand_resname_list=input_parameters['kept_ligand_resname_list'],
                                           reference_pdb_file_name_for_alignment=input_parameters['reference_pdb_file_name_for_alignment'],
                                           working_dir_name=input_parameters['working_dir'])

        prepared_protein_pdb_file_name_list, prepared_protein_conf_name_list = protein_fixer.run()
        num_protein_conformations = len(prepared_protein_conf_name_list)
        built_protein_pdb_file_name_list = [None] * num_protein_conformations

        if num_protein_conformations == 0:
            return 0

        for protein_conf_idx in range(num_protein_conformations):
            current_prepared_protein_pdb_file_name = prepared_protein_pdb_file_name_list[protein_conf_idx]
            current_prepared_protein_conf_name = prepared_protein_conf_name_list[protein_conf_idx]
            current_protein_system_builder_runner = ProteinSystemBuilderRunner(protein_pdb_file_name=current_prepared_protein_pdb_file_name,
                                                                               protein_conf_name=current_prepared_protein_conf_name,
                                                                               kept_ligand_resname_list=input_parameters['kept_ligand_resname_list'],
                                                                               working_dir_name=input_parameters['working_dir'])

            built_protein_pdb_file_name_list[protein_conf_idx] = current_protein_system_builder_runner.run()

        returned_protein_info_dict = {}
        returned_protein_info_dict['protein_conf_name'] = prepared_protein_conf_name_list
        returned_protein_info_dict['protein_pdb_file_name'] = built_protein_pdb_file_name_list
        returned_protein_info_df = pd.DataFrame(returned_protein_info_dict)

        returned_protein_info_csv_file_name = os.path.join(input_parameters['working_dir'], 'built_protein_models.csv')
        returned_protein_info_df.to_csv(returned_protein_info_csv_file_name, index=False)
