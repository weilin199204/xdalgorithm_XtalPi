import os
import copy


def get_key_from_dict(input_dict, query_key):
    if query_key not in input_dict:
        raise KeyError('{0} is not found in molgen parameters.')
    return input_dict[query_key]


class TransferLearning(object):
    def __init__(self, configuration):
        self.configuration = configuration

    def run(self):
        from xdalgorithm.toolbox.reinvent.running_modes.manager import Manager as ReinventManager
        from xdalgorithm.toolbox.reinvent import process_tools
        trainset_path = get_key_from_dict(get_key_from_dict(self.configuration, 'parameters'),
                                          'input_smiles_path')
        prep_trainset_path = os.path.basename(trainset_path) + '.prep'
        prep_error_path = os.path.basename(trainset_path) + ".error"

        process_tools.preprocessing(trainset_path, prep_trainset_path, prep_error_path)
        self.configuration['parameters']['input_smiles_path'] = prep_trainset_path

        manager = ReinventManager(self.configuration)
        manager.run()


class Sampling(object):
    def __init__(self, configuration):
        self.configuration = configuration

    def run(self):
        from xdalgorithm.toolbox.reinvent.running_modes.manager import Manager as ReinventManager
        from xdalgorithm.toolbox.reinvent import process_tools
        model_path = get_key_from_dict(get_key_from_dict(self.configuration, 'parameters'), 'model_path')
        if isinstance(model_path, list):
            # save raw smiles sampled from model list in logging
            model_list = model_path
            # for every
            original_output_smiles_path = self.configuration['parameters']['output_smiles_path']
            # raw_output_smiles_path = os.path.join(configuration['logging']['logging_path'],"sampling_smiles.raw")
            if not os.path.exists(self.configuration['logging']['logging_path']):
                os.mkdir(self.configuration['logging']['logging_path'])
            # collect sampling dataset paths
            sampling_file_paths = []

            for index, model_path in enumerate(model_list):
                # modify the model path
                self.configuration['parameters']['model_path'] = model_path
                index_smi_file_name = os.path.join(self.configuration['logging']['logging_path'],
                                                   "sampling_{0}_{1}.raw".format(index, os.path.basename(model_path)))
                # get a new configuration
                self.configuration['parameters']['output_smiles_path'] = index_smi_file_name
                manager = ReinventManager(self.configuration)
                manager.run()

                sampling_file_paths.append(index_smi_file_name)
            # merge all raw smiles in a file
            process_tools.postprocessing(sampling_file_paths, original_output_smiles_path)
        else:
            # print('sampling else')
            # replace the output_smiles_path,save raw file in log folder
            original_output_smiles_path = self.configuration['parameters']['output_smiles_path']
            raw_output_smiles_path = os.path.join(self.configuration['logging']['logging_path'], "sampling_smiles.raw")
            if not os.path.exists(self.configuration['logging']['logging_path']):
                os.mkdir(self.configuration['logging']['logging_path'])

            self.configuration['parameters']['output_smiles_path'] = raw_output_smiles_path
            manager = ReinventManager(self.configuration)
            manager.run()
            # save postprocessed result in input_output_smiles_path
            process_tools.postprocessing([raw_output_smiles_path], original_output_smiles_path)


class ReinforcementLearning:

    def __init__(self, configuration):
        self.configuration = configuration

    def run(self):
        from xdalgorithm.toolbox.reinvent.running_modes.manager import Manager as ReinventManager
        manager = ReinventManager(self.configuration)
        manager.run()


class ModelCreation(object):
    def __init__(self, configuration):
        self.configuration = configuration

    def run(self):
        from xdalgorithm.toolbox.reinvent.running_modes.manager import Manager as ReinventManager
        # The configuration to create is divided into create empty model configuration and transfer
        # learning configuration,they share the same logging path.Actually,log of configuration to create empty model
        # is empty.
        logging_dict = get_key_from_dict(self.configuration, 'logging')
        create_model_configuration = {
            'version': self.configuration['version'],
            'run_type': 'create_model',
            'parameters': get_key_from_dict(self.configuration, 'create_parameters'),
            'logging': {
                'sender': get_key_from_dict(logging_dict, 'sender'),
                'recipient': get_key_from_dict(logging_dict, 'recipient'),
                'logging_path': get_key_from_dict(logging_dict, 'logging_path'),
                'job_name': get_key_from_dict(logging_dict, 'job_name')
            }
        }

        tl_configuration = {
            'version': get_key_from_dict(self.configuration, 'version'),
            'run_type': 'transfer_learning',
            'parameters': get_key_from_dict(self.configuration, 'tl_parameters'),
            'logging': {
                'sender': get_key_from_dict(logging_dict, 'sender'),
                'recipient': get_key_from_dict(logging_dict, 'recipient'),
                'logging_path': get_key_from_dict(logging_dict, 'logging_path'),
                'job_name': get_key_from_dict(logging_dict, 'job_name'),
                'use_weights': get_key_from_dict(logging_dict, 'use_weights'),
                'job_id': get_key_from_dict(logging_dict, 'job_id')
            }
        }
        # firstly,create a empty model
        manager = ReinventManager(create_model_configuration)
        manager.run()
        # CLICommand.run_test_create(tl_configuration)
        # secondly,train the model.
        manager = ReinventManager(tl_configuration)
        manager.run()


class CLICommand:
    """A machine learning tool for molecular generations.

    Approaches:
    Using recurrent neural network to learn the molecular generation through:
    - Transfer learning
    - Reinforcement learning
    -------------------------------------------------------------------------
    Input json for create model:
    {
      "version": "2.0",
      "run_type": "create_model",
      "create_parameters": {
        "output_model_path": "path/to/empty_model.prior",
        "input_smiles_path": "demo_dataset.smi",
        "num_layers": 3,
        "layer_size": 512,
        "cell_type": "lstm",
        "embedding_layer_size": 256,
        "dropout": 0,
        "max_sequence_length": 256,
        "layer_normalization": false,
        "standardize": 0
      },
      "tl_parameters": {
        "input_model_path": "path/to/empty_model.prior",
        "output_model_path": "path/to/created_model.prior",
        "input_smiles_path": "path/to/demo_dataset.smi",
        "save_every_n_epochs": 1,
        "batch_size": 128,
        "clip_gradient_norm": 1.0,
        "num_epochs": 5,
        "starting_epoch": 1,
        "shuffle_each_epoch": 1,
        "collect_stats_frequency": 1,
        "standardize": true,
        "randomize": true,
        "max_heavy_atoms": 70,
        "min_heavy_atoms": 2,
        "adaptive_lr_config": {
          "mode": "constant",
          "gamma": 0.8,
          "step": 1,
          "start": 5E-4,
          "min": 1E-5,
          "threshold": 1E-4,
          "average_steps": 4,
          "patience": 8,
          "restart_value": 1E-5,
          "sample_size": 100,
          "restart_times": 0
        }
      },
      "logging": {
        "sender": "this is relevant for remote job submission scenario",
        "recipient": "local",
        "logging_path": "path/of/create_demo.log",
        "job_name": "new_model",
        "use_weights": 0,
        "job_id": "job_id"
      }
    }
    -------------------------------------------------------------------------
    Input json for transfer learning:
    {
      "version": "2.0",
      "run_type": "transfer_learning",
      "parameters": {
        "input_model_path": "augmented.prior",
        "output_model_path": "focused.ckpt",
        "input_smiles_path": "smiles.smi",
        "save_every_n_epochs": 1,
        "batch_size": 128,
        "clip_gradient_norm": 1.0,
        "num_epochs": 5,
        "starting_epoch": 1,
        "shuffle_each_epoch": 1,
        "collect_stats_frequency": 1,
        "standardize": true,
        "randomize": true,
        "max_heavy_atoms": 70,
        "min_heavy_atoms": 2,

        "adaptive_lr_config": {
          "mode": "constant",
          "gamma": 0.8,
          "step": 1,
          "start": 5E-4,
          "min": 1E-5,
          "threshold": 1E-4,
          "average_steps": 4,
          "patience": 8,
          "restart_value": 1E-5,
          "sample_size": 100,
          "restart_times": 0
        }
      },
      "logging": {
        "sender": "this is relevant for remote job submission scenario",
        "recipient": "local",
        "logging_path": "/path/to/a/logging/folder/progress.log",
        "use_weights": 0,
        "job_name": "transfer learning",
        "job_id": "relevant for remote logging"
      }
    }
    ----------------------------------------------------------------------
    Input json template for reinforcement learning:
    {
      "version": "2.0",
        "run_type": "reinforcement_learning",
        "parameters": {
            "scoring_function": {
                "name": "custom_sum",
                "parallel": true,
                "parameters": [
                    {
                        "component_type": "tpsa",
                        "name": "TPSA",
                        "weight": 1,
                        "model_path": null,
                        "smiles": [],
                        "specific_parameters": {
                            "transformation_type": "double_sigmoid",
                            "high": 140,
                            "low": 0,
                            "coef_div": 140,
                            "coef_si": 20,
                            "coef_se": 20,
                            "transformation": true
                        }
                    },
                    {
                        "component_type": "num_rotatable_bonds",
                        "name": "Number of rotatable bonds",
                        "weight": 2,
                        "model_path": null,
                        "smiles": [],
                        "specific_parameters": {
                            "transformation_type": "step",
                            "high": 8,
                            "low": 3,
                            "transformation": true
                        }
                    },
                    {
                        "component_type": "num_hbd_lipinski",
                        "name": "HB-donors (Lipinski)",
                        "weight": 2,
                        "model_path": null,
                        "smiles": [],
                        "specific_parameters": {
                            "transformation_type": "step",
                            "high": 5,
                            "low": 1,
                            "transformation": true
                        }
                    },
                    {
                        "component_type": "num_rings",
                        "name": "Number of rings",
                        "weight": 2,
                        "model_path": null,
                        "smiles": [],
                        "specific_parameters": {
                            "transformation_type": "step",
                            "high": 7,
                            "low": 2,
                            "transformation": true
                        }
                    }
                ]
            },
            "diversity_filter": {
                "name": "IdenticalMurckoScaffold",
                "nbmax": 32,
                "minscore": 0.55,
                "minsimilarity": 0.5
            },
            "reinforcement_learning": {
                "prior": "model.ckpt.pri",
                "agent": "model.ckpt.agent",
                "n_steps": 10,
                "sigma": 128,
                "learning_rate": 0.0001,
                "batch_size": 64,
                "reset": 0,
                "reset_score_cutoff": 0.5,
                "margin_threshold": 50
            },
            "inception": {
                "smiles": [],
                "memory_size": 512,
                "sample_size": 256
            }
        },
        "logging": {
            "sender": "this is relevant for remote job submission scenario",
            "recipient": "local",
            "logging_frequency": 4,
            "logging_path": "progress.log",
            "resultdir": "./results/directory",
            "job_name": "Reinforcement learning",
            "job_id": "relevant for remote logging"
        }
    }

    ------------------------------------------------------------------------------
    Input json for sampling:
    {
      "version": "2.0",
      "run_type": "sampling",
      "parameters": {
        "model_path": "path/to/generative/model/to/sample/from/augmented.prior",
        "output_smiles_path": "path/to/store.the/sampled/smiles/sampled.smi",
        "num_smiles": 1024,
        "batch_size": 128,
        "with_likelihood": false
      },
      "logging": {
        "sender": "this is relevant for remote job submission scenario",
        "recipient": "local",
        "logging_path": "/path/to/a/logging/folder/progress.log",
        "job_name": "sampling"
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
        # if 'start_smiles' not in configuration:
        #     configuration['start_smiles'] = ''
        input_configuration = copy.deepcopy(configuration)
        runner_dictionary = {
            "sampling": Sampling,
            "transfer_learning": TransferLearning,
            "reinforcement_learning": ReinforcementLearning,
            "create_model": ModelCreation
        }
        if configuration['run_type'] not in runner_dictionary:
            raise ValueError('Unknown run_type {0} is found!'.format(configuration['run_type']))
        current_runner = runner_dictionary.get(configuration['run_type'])(configuration)
        current_runner.run()
        save_arguments_to_json(input_configuration,
                               os.path.join(input_configuration['logging']['logging_path'], 'input.json'))
