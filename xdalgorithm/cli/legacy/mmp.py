class CLICommand:
    """MMP-rule based molecular design (Modifier)

    The main idea behind is that fragments in the identical context are interchangeable.
    Therefore, one can create a database of interchangeable fragments and use it for
    generation of chemically valid structures.

    **Features:**
    1) Three modes of structure generation: GROW, MUTATE, LINK
    2) Context radius to consider for replacement
    3) Fragment size to replace and the size of a replacing fragment
    4) Protection of atoms from modification (e.g. scaffold protection)
    5) Replacements with fragments occurred in a fragment database with certain minimal frequency
    6) Make randomly chosen replacements up to the specified number

    -----------------------------------------------------------------------------------------------
    Input json template for GROW:
    {
        "logging_path": "logging.log",
        "max_atoms": 2,
        "max_replacements": null,
        "min_atoms": 1,
        "min_freq": 0,
        "mol": "c1cc(OC)ccc1C",
        "ncores": 1,
        "output_smiles_path": "grow_output.smi",
        "protected_ids": null,
        "radius": 3,
        "replace_ids": null,
        "run_type": "mmp_grow"
    }
    ------------------------------------------------------------------------------------------------
    Input json template for MUTATE:
    {
        "logging_path": "logging.log",
        "max_inc": 2,
        "max_rel_size": 10,
        "max_replacements": null,
        "max_size": 10,
        "min_freq": 0,
        "min_inc": -2,
        "min_rel_size": 0,
        "min_size": 0,
        "mol": "c1cc(OC)ccc1C",
        "ncores": 1,
        "output_smiles_path": "mutate_output.smi",
        "protected_ids": false,
        "radius": 3,
        "replace_cycles": false,
        "replace_ids": null,
        "run_type": "mmp_mutate"
    }
    -----------------------------------------------------------------------------
    Input json template for LINK:
    {
        "run_type": "mmp_link", 
        "mol_1": "COc1ccc2c(c1)NCCC2", 
        "mol_2": "C[C@@H]1CC(=O)Nc2ccccc2N1", 
        "logging_path": "logging.log",
        "output_smiles_path": "link_output.smi",
        "radius": 1, 
        "min_atoms": 6, 
        "max_atoms": 10, 
        "max_replacements": null, 
        "min_freq": 0, 
        "ncores": 1, 
        "dist": [4, 6], 
        "replace_ids_1": [8], 
        "replace_ids_2": [10], 
        "protected_ids_1": null, 
        "protected_ids_2": null
    }
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-f', '--input-json', default='', type=str)

    @staticmethod
    def run(args):
        from xdalgorithm.utils import load_arguments_from_json
        from xdalgorithm.toolbox.crem.crem_manager import CremManager
        configuration = load_arguments_from_json(args.input_json)
        crem_manager = CremManager(configuration)
        crem_manager.run()
