from xdalgorithm.toolbox.reinvent.running_modes.transfer_learning.logging.neptune_transfer_learning_logger import NeptuneTransferLearningLogger
from xdalgorithm.toolbox.reinvent.running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from xdalgorithm.toolbox.reinvent.running_modes.configurations.logging.reinforcement_log_configuration import ReinforcementLoggerConfiguration
from xdalgorithm.toolbox.reinvent.running_modes.reinforcement_learning.logging.base_reinforcement_logger import BaseReinforcementLogger
from xdalgorithm.toolbox.reinvent.running_modes.reinforcement_learning.logging import LocalReinforcementLogger
from xdalgorithm.toolbox.reinvent.running_modes.reinforcement_learning.logging.remote_reinforcement_logger import RemoteReinforcementLogger
from xdalgorithm.toolbox.reinvent.running_modes.reinforcement_learning.logging.neptune_reinforcement_logger import NeptuneReinforcementLogger
from xdalgorithm.toolbox.reinvent.utils.enums.logging_mode_enum import LoggingModeEnum


class ReinforcementLogger:

    def __new__(cls, configuration: GeneralConfigurationEnvelope) -> BaseReinforcementLogger:
        logging_mode_enum = LoggingModeEnum()
        rl_config = ReinforcementLoggerConfiguration(**configuration.logging)
        if rl_config.recipient == logging_mode_enum.LOCAL:
            logger_instance = LocalReinforcementLogger(configuration)
        elif rl_config.recipient == logging_mode_enum.REMOTE:
            logger_instance = RemoteReinforcementLogger(configuration)
        elif rl_config.recipient == logging_mode_enum.NEPTUNE:
            logger_instance = NeptuneReinforcementLogger(configuration)
        else:
            print("Not logger specified.")

        return logger_instance
