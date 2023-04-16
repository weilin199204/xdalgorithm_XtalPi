class CLICommand:
    """A toolkit to do synthetic route analysis.

    Input json template for rebuild:
    {
        "run_type": "xreact_rebuild", 
        "bb_dict": 
        {
            "aryl_vinyl_halides/suzuki.smi": "input/filtering.smi",
            "primary_secondary_amines/amination-1.smi": "rep_mols.smi"
        }, 
	    "core_smiles": "c1ncc2nc[nH]c2n1", 
	    "route": 
        ["suzuki", 
		    "NS(=O)(=O)c1ccc(Nc2nc(c3cccc(c3)c4ccccc4)c5nc[nH]c5n2)cc1", "Brc1cccc(-c2ccccc2)c1", 
			[
				"amination-1", 
				"NS(=O)(=O)c1ccc(Nc2nc(B(O)O)c3nc[nH]c3n2)cc1", 
				"OB(O)c1nc(Br)nc2[nH]cnc12", 
				"Nc1ccc(S(N)(=O)=O)cc1"
            ]
		], 
		"logging_path": "rebuild_logging.log", 
		"output_path": "output/smiles/file/path.smi",
		"db_path": "/data/aidd-server/xreact_reactions"
    }
    -------------------------------------------------------------------------------------------
    Input json template for clustering:
    {
        "run_type": "xreact_clustering", 
        "reactant_def": "primary_secondary_amines/amination-1.smi", 
        "bb_from": "/compounds.smi",
        "logging_path": "clustering.log", 
        "rep": "least_num_atoms", 
        "cutoff": 0.8, 
        "db_path": "/data/aidd-server/xreact_reactions"
    }
    -------------------------------------------------------------------------------------------
    Input json template for retrosynthesis
    {
        "run_type": "xreact_retrosynthesis",
        "start_smiles": "NS(=O)(=O)c1ccc(Nc2nc(c3cccc(c3)c4ccccc4)c5nc[nH]c5n2)cc1",
        "core_smiles": "c1ncc2nc[nH]c2n1",
        "core_single_reactive_center": true,
        "core_specific": true,
        "db_path": "/data/aidd-server/xreact_reactions",
        "logging_path": "/logging/diretory/saving/json/imgs"
    }
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-f', '--input-json', default='', type=str)

    @staticmethod
    def run(args):
        from xdalgorithm.utils import load_arguments_from_json
        from xdalgorithm.toolbox.xreact.xreact_manager import XreactManager
        configuration = load_arguments_from_json(args.input_json)
        xreact_manager = XreactManager(configuration)
        xreact_manager.run()
