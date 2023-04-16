from xdalgorithm.toolbox.reinvent.running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from xdalgorithm.toolbox.reinvent.running_modes.configurations.logging.sampling_log_configuration import SamplingLoggerConfiguration
from xdalgorithm.toolbox.reinvent.running_modes.sampling.logging.base_sampling_logger import BaseSamplingLogger
from xdalgorithm.toolbox.reinvent.running_modes.sampling.logging.local_sampling_logger import LocalSamplingLogger
from xdalgorithm.toolbox.reinvent.running_modes.sampling.logging.remote_sampling_logger import RemoteSamplingLogger
from xdalgorithm.toolbox.reinvent.utils.enums.logging_mode_enum import LoggingModeEnum


class SamplingLogger:
    def __new__(cls, configuration: GeneralConfigurationEnvelope) -> BaseSamplingLogger:
        logging_mode_enum = LoggingModeEnum()
        sampling_config = SamplingLoggerConfiguration(**configuration.logging)
        if sampling_config.recipient == logging_mode_enum.LOCAL:
            logger = LocalSamplingLogger(configuration)
        else:
            logger = RemoteSamplingLogger(configuration)
        return logger