import os

class CLICommand:
    __doc__ = "A protein-ligand topology and FFPs building tool for generating inputs of MD simulations.\n" +\
    """ Input json template:
    {
        "parameters":
        {
            "protein_pdb_file_name": "/protein_path.pdb",
            "ligand_sdf_file_name": "/ligand_path.sdf",
            "ligand_resname": "MOL",
            "water_box_shape": "octahedral",
            "water_box_edge": 1.0,
            "ff_format": "AMBER",
            "protein_prmtop_file_name": "protein.prmtop",
            "protein_inpcrd_file_name": "protein.inpcrd",
            "ligand_prmtop_file_name": "ligand.prmtop",
            "ligand_inpcrd_file_name": "ligand.inpcrd",
            "system_unsolvated_prmtop_file_name": "system_unsolvated.prmtop",
            "system_unsolvated_inpcrd_file_name": "system_unsolvated.inpcrd",
            "system_unsolvated_psf_file_name": "system_unsolvated.psf",
            "system_unsolvated_pdb_file_name": "system_unsolvated.pdb",
            "system_unsolvated_top_file_name": "system_unsolvated.top",
            "system_unsolvated_gro_file_name": "system_unsolvated.gro",
            "system_solvated_prmtop_file_name": "system_solvated.prmtop",
            "system_solvated_inpcrd_file_name": "system_solvated.inpcrd",
            "system_solvated_psf_file_name": "system_solvated.psf",
            "system_solvated_pdb_file_name": "system_solvated.pdb",
            "system_solvated_top_file_name": "system_solvated.top",
            "system_solvated_gro_file_name": "system_solvated.gro",
            "working_dir_name": "."
        }
    }
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-f', '--input-json', default='', type=str)

    @staticmethod
    def run(args):
        from xdalgorithm.utils import load_arguments_from_json
        from xdalgorithm.toolbox.md.system_building.md_system_builder_runner import MDSystemBuilderRunner
        configuration = load_arguments_from_json(args.input_json)

        work_dir = configuration.get('parameters', ValueError).get('work_dir', '.')
        if not os.path.exists(work_dir) or not os.path.isdir(work_dir):
            os.mkdir(work_dir)

        function_dictionary = {'md_system_building': MDSystemBuilderRunner}
        input_parmaters = configuration['parameters']
        current_job = function_dictionary.get(configuration['run_type'], ValueError)(**input_parmaters)
        current_job.run()
