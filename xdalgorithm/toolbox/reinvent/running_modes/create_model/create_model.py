#!/usr/bin/env python
#  coding=utf-8

import xdalgorithm.toolbox.reinvent.models.model as reinvent
import xdalgorithm.toolbox.reinvent.models.vocabulary as voc
from xdalgorithm.toolbox.reinvent.running_modes.configurations.logging.create_model_log_configuration import CreateModelLoggerConfiguration
from xdalgorithm.toolbox.reinvent.running_modes.create_model.logging.create_model_logger import CreateModelLogger
from xdalgorithm.toolbox.reinvent.running_modes.create_model.logging.remote_create_model_logger import RemoteCreateModelLogger
from xdalgorithm.toolbox.reinvent.running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from xdalgorithm.toolbox.reinvent.running_modes.configurations.create_model.create_model_configuration import CreateModelConfiguration
import xdalgorithm.toolbox.reinvent.utils.smiles as chem_smiles
from xdalgorithm.toolbox.reinvent.utils.enums.logging_mode_enum import LoggingModeEnum


class CreateModelRunner:

    def __init__(self, main_config: GeneralConfigurationEnvelope, configuration: CreateModelConfiguration):
        """
        Creates a CreateModelRunner.
        """
        self._smiles_list = chem_smiles.read_smiles_file(configuration.input_smiles_path, standardize=configuration.standardize)
        self._output_model_path = configuration.output_model_path

        self.start_smiles = configuration.start_smiles

        self._num_layers = configuration.num_layers
        self._layer_size = configuration.layer_size
        self._cell_type = configuration.cell_type
        self._embedding_layer_size = configuration.embedding_layer_size
        self._dropout = configuration.dropout
        self._max_sequence_length = configuration.max_sequence_length
        self._layer_normalization = configuration.layer_normalization
        self.logger = self._resolve_logger(main_config)

    def run(self):
        """
        Carries out the creation of the model.
        """

        tokenizer = voc.SMILESTokenizer()
        vocabulary = voc.create_vocabulary(self._smiles_list, tokenizer=tokenizer)

        network_params = {
            'num_layers': self._num_layers,
            'layer_size': self._layer_size,
            'cell_type': self._cell_type,
            'embedding_layer_size': self._embedding_layer_size,
            'dropout': self._dropout,
            'layer_normalization': self._layer_normalization,
            # 'start_smiles': self.start_smiles
        }
        model = reinvent.Model(
            no_cuda=True,
            vocabulary=vocabulary,
            tokenizer=tokenizer,
            network_params=network_params,
            max_sequence_length=self._max_sequence_length,
            start_smiles=self.start_smiles
        )
        model.save(self._output_model_path)
        return model

    def _resolve_logger(self, configuration: GeneralConfigurationEnvelope):
        logging_mode_enum = LoggingModeEnum()
        create_model_config = CreateModelLoggerConfiguration(**configuration.logging)
        if create_model_config.recipient == logging_mode_enum.LOCAL:
            logger = CreateModelLogger(configuration)
        else:
            logger = RemoteCreateModelLogger(configuration)
        return logger

