from xdalgorithm.toolbox.reinvent.running_modes.configurations.general_configuration_envelope import GeneralConfigurationEnvelope
from xdalgorithm.toolbox.reinvent.running_modes.configurations.scoring.scoring_runner_configuration import ScoringRunnerConfiguration
from xdalgorithm.toolbox.reinvent.running_modes.scoring.logging.scoring_logger import ScoringLogger
from xdalgorithm.toolbox.reinvent.scoring.function.base_scoring_function import BaseScoringFunction
from xdalgorithm.toolbox.reinvent.scoring.score_summary import FinalSummary
from xdalgorithm.toolbox.reinvent.utils.smiles import read_smiles_file


class ScoringRunner:
    def __init__(self, configuration: GeneralConfigurationEnvelope, config: ScoringRunnerConfiguration,
                 scoring_function: BaseScoringFunction):
        self._scoring_function = scoring_function
        self._config = config
        self._logger = ScoringLogger(configuration)

    def run(self):
        input_smiles = list(read_smiles_file(file_path=self._config.input, randomize=False, standardize=False))
        score_summary: FinalSummary = self._scoring_function.get_final_score(input_smiles)

        self._logger.log_results(score_summary=score_summary)
        self._logger.log_out_input_configuration()
