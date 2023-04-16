import os
import copy


class Filtering:

    def __init__(self, configuration):
        self.configuration = configuration

    def run(self):
        from xdalgorithm.toolbox.virtual_screening.rand_id import get_rand_id
        from xdalgorithm.toolbox.scoring_filtering.scoring_filtering_manager import \
            Scoring_Filtering_Manager
        if isinstance(self.configuration['parameters']['scoring']['input'], list):
            all_smiles = ""
            for input_path in self.configuration['parameters']['scoring']['input']:
                smiles = open(input_path, "r").read()
                all_smiles += smiles
                if not all_smiles.endswith("\n"):
                    all_smiles += "\n"
            input_smi = "./filtering-{0}".format(get_rand_id())
            with open(input_smi, "w") as fw:
                fw.write(all_smiles)
            self.configuration['parameters']['scoring']['input'] = os.path.abspath(input_smi)

        manager = Scoring_Filtering_Manager(self.configuration)
        manager.run()


class Scoring:

    def __init__(self, configuration):
        self.configuration = configuration

    def run(self):
        from xdalgorithm.utils import get_rand_id
        from xdalgorithm.toolbox.scoring_filtering.scoring_filtering_manager import \
            Scoring_Filtering_Manager
        if isinstance(self.configuration['parameters']['scoring']['input'], list):
            all_smiles = ""
            for input_path in self.configuration['parameters']['scoring']['input']:
                smiles = open(input_path, "r").read()
                all_smiles += smiles
                if not all_smiles.endswith("\n"):
                    all_smiles += "\n"
            input_smi = "./scoring-{0}".format(get_rand_id())
            with open(input_smi, "w") as fw:
                fw.write(all_smiles)
            self.configuration['parameters']['scoring']['input'] = os.path.abspath(input_smi)
        manager = Scoring_Filtering_Manager(self.configuration)
        manager.run()


class CLICommand:
    """The scoring and filtering module implementing some physicochemical properties for small molecules.
        Calculation of properties for compounds

        Two protocols are supported: filtering and scoring.

        Input json template for filtering:
        {
            "version": 2,
            "run_type": "filtering",
            "logging": {
                "sender": "this is relevant for remote job submission scenario",
                "recipient": "local",
                "logging_path": "/directory/path/to/save/output/csv/file.log",
                "job_name": "scoring",
                "job_id": "relevant for remote logging"
            },
            "parameters": {
                "scoring_function":{
                    "name": "custom_filtering",
                    "parallel": false,
                    "parameters":
                        [
                            {
                                "component_type": "tpsa",
                                "name": "drug_tpsa",
                                "weight": 1,
                                "low_limit": [">",100],
                                "up_limit": ["<",400]
                            },
                            {
                                "component_type": "tanimoto_similarity",
                                "smiles": ["c1ccccc1CC", "c1ccccc1CCC", "c1ccccc1CCCC"],
                                "name": "drug_similarity",
                                "low_limit": [">",0.1]
                            },
                            {
                                "component_type": "molecular_weight",
                                "model_path": null,
                                "smiles": [],
                                "name": "molecular weight",
                                "up_limit": ["<",500]
                            },
                            {
                                "component_type":"pains",
                                "name":"pains",
                                "smiles":[]
                            }
                        ]
                },

            "scoring": {
                "input": "/smiles/file/path/to/calculate/properties.smi"
                }
            }
        }

        Input json template for scoring:
        {
          "version": 2,
          "run_type": "scoring",
          "logging": {
                "sender": "this is relevant for remote job submission scenario",
                "recipient": "local",
                "logging_path": "/directory/path/to/save/output/csv/file.log",
                "job_name": "scoring",
                "job_id": "relevant for remote logging"
            },
          "parameters": {
                "scoring_function":
                {
                    "name": "custom_filtering",
                    "parallel": false,
                    "parameters":[
                        {
                            "component_type": "tpsa",
                            "model_path": null,
                            "smiles": [],
                            "name": "tpsa",
                            "low_limit": [">",50],
                            "up_limit": ["<",90]
                        },
                        {
                            "component_type": "tanimoto_similarity",
                            "smiles": ["c1ccccc1CC", "c1ccccc1CCC", "c1ccccc1CCCC"],
                            "name": "Tanimoto Similarity",
                            "low_limit": [">",0.1]
                        },
                        {
                            "component_type": "molecular_weight",
                            "model_path": null,
                            "smiles": [],
                            "name": "molecular weight",
                            "up_limit": ["<",500]
                        },
                        {
                            "component_type":"custom_alerts",
                            "name":"custom_alerts",
                            "smiles":[
                                  "c1cccnc1"
                            ]
                        },
                        {
                            "component_type": "qed_score",
                            "name": "QED Score"
                        },
                        {
                            "component_type": "matching_substructure",
                            "name": "Matching substructure",
                            "smiles": [
                                "c1ccccc1CC"
                            ]
                        },
                        {
                            "component_type": "jaccard_distance",
                            "name": "Jaccard dissimilarity",
                            "smiles": [
                                     "CCCCCCCCC"
                            ]
                        },
                        {
                            "component_type": "num_rotatable_bonds",
                            "name": "Number of rotatable bonds"
                        },
                        {
                            "component_type": "num_hbd_lipinski",
                            "name": "HB-donors (Lipinski)"
                        },
                        {
                            "component_type": "num_rings",
                            "name": "Number of rings"
                        },
                        {
                            "component_type": "sa_score",
                            "name": "SA Score",
                            "model_path": null
                        },
                        {
                            "component_type":"pains",
                            "name":"pains",
                            "specific_parameters":{
                                "tautomer_query":true
                            }
                        },
                        {
                            "component_type":"ld50_oral",
                            "name":"ld50_oral"
                        },
                       {
                            "component_type":"nonbiodegradable",
                            "name":"nonbiodegradable"
                        },
                        {
                            "component_type":"acute_aquatic_toxicity",
                            "name":"acute_aquatic_toxicity"
                        },
                        {
                            "component_type":"aggregators",
                            "name":"aggregators"
                        },
                        {
                            "component_type":"alphascreen_gst_fhs",
                            "name":"alphascreen_gst_fhs"
                        },
                        {
                            "component_type":"custom_component",
                            "name":"custom_component",
                            "specific_parameters":{
                            "function":"from rdkit.Chem import Descriptors\ndef calculate_property(m):\n    tpsa_m = Descriptors.TPSA(m)\n    logp_m = Descriptors.MolLogP(m)\n    return tpsa_m+logp_m"
                            }
                        }
                    ]
                },

                "scoring": {
                    "input": "/smiles/file/path/to/calculate/properties.smi"
                }
            }
        }

        Commments:
            1. Some commponent_types related to substructure matching support tautomer_query, which is a new function in rdkit 2020.09

            2. Scoring json can not be used in molgen reinforcement learning

            3. scoring functions from reinforcement learning is supported in screen.

            4. sa_score component is allowed to receive a sckit-learn model to predict SA.If model is none, inner model is used.

            5. list of support properties:
                matching_substructure,tanimoto_similarity,jaccard_distance,custom_alerts,qed_score,
                molecular_weight,tpsa,num_rotatable_bonds,num_hbd_lipinski,num_rings,sa_score,custom_component,
                ld50_oral,acute_aquatic_toxicity,aggregators,alphascreen_gst_fhs,alarm_nmr,
                alphascreen_fhs,alphascreen_his_fhs,bms,developmental_mitochondrial,chelating,
                dna_binding,reactive_unstable_toxic,extended_functional_groups,frequent_hitters,function_group,
                genotoxic_carcinogenicity_mutagenicity,idiosyncratic,luciferase_inhibitory,ntd,nonbiodegradable,
                nongenotoxic_carcinogenicity,pains,potential_electrophilic,biodegradable,skin_sensitization,
                surechembl,toxicophores
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-f', '--input-json', default='', type=str)

    @staticmethod
    def run(args):
        from xdalgorithm.utils import load_arguments_from_json, save_arguments_to_json
        configuration = load_arguments_from_json(args.input_json)
        input_configuration = copy.deepcopy(configuration)
        runner_dictionary = {
            "scoring": Scoring,
            "filtering": Filtering,
        }
        if configuration['run_type'] not in runner_dictionary:
            raise ValueError('Unknown run_type {0} is found!'.format(configuration['run_type']))
        current_runner = runner_dictionary.get(configuration['run_type'])(configuration)
        current_runner.run()
        save_arguments_to_json(input_configuration,
                               os.path.join(input_configuration['logging']['logging_path'], 'input.json'))
