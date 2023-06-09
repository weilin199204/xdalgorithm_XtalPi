from xdalgorithm.toolbox.reinvent.running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from xdalgorithm.toolbox.reinvent.running_modes.configurations.logging.transfer_learning_log_configuration import TransferLearningLoggerConfig
from xdalgorithm.toolbox.reinvent.running_modes.transfer_learning.logging.base_transfer_learning_logger import BaseTransferLearningLogger
from xdalgorithm.toolbox.reinvent.running_modes.transfer_learning.logging.local_transfer_learning_logger import LocalTransferLearningLogger
from xdalgorithm.toolbox.reinvent.running_modes.transfer_learning.logging.remote_transfer_learning_logger import RemoteTransferLearningLogger
from xdalgorithm.toolbox.reinvent.running_modes.transfer_learning.logging.neptune_transfer_learning_logger import NeptuneTransferLearningLogger
from xdalgorithm.toolbox.reinvent.utils.enums.logging_mode_enum import LoggingModeEnum


class TransferLearningLogger:
    def __new__(cls, configuration: GeneralConfigurationEnvelope) -> BaseTransferLearningLogger:
        logging_mode_enum = LoggingModeEnum()
        tl_config = TransferLearningLoggerConfig(**configuration.logging)
        if tl_config.recipient == logging_mode_enum.LOCAL:
            logger = LocalTransferLearningLogger(configuration)
        elif tl_config.recipient == logging_mode_enum.REMOTE:
            logger = RemoteTransferLearningLogger(configuration)
        elif tl_config.recipient == logging_mode_enum.NEPTUNE:
            logger = NeptuneTransferLearningLogger(configuration)
        else:
            print("Not logger specified.")

        return logger