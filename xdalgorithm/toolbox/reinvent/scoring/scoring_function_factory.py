from xdalgorithm.toolbox.reinvent.scoring.component_parameters import ComponentParameters
from xdalgorithm.toolbox.reinvent.scoring.function import CustomProduct, CustomSum # CustomFiltering
from xdalgorithm.toolbox.reinvent.scoring.function.base_scoring_function import BaseScoringFunction
from xdalgorithm.toolbox.reinvent.scoring.scoring_function_parameters import ScoringFuncionParameters
from xdalgorithm.toolbox.reinvent.utils.enums.scoring_function_enum import ScoringFunctionNameEnum

#ScoringFunctionParameters
#        self.name = name
#        self.parameters = parameters
#        self.parallel = parallel

class ScoringFunctionFactory:

    def __new__(cls, sf_parameters: ScoringFuncionParameters) -> BaseScoringFunction:
        enum = ScoringFunctionNameEnum()
        scoring_function_registry = {
            enum.CUSTOM_PRODUCT: CustomProduct,
            enum.CUSTOM_SUM: CustomSum
            # enum.CUSTOM_FILTERING: CustomFiltering
        }
        return cls.create_scoring_function_instance(sf_parameters, scoring_function_registry)

    @staticmethod
    def create_scoring_function_instance(sf_parameters: ScoringFuncionParameters,
                                         scoring_function_registry: dict) -> BaseScoringFunction:
        """Returns a scoring function instance"""
        scoring_function = scoring_function_registry[sf_parameters.name]
        parameters = [ComponentParameters(**p) for p in sf_parameters.parameters]

        return scoring_function(
            parameters,
            sf_parameters.parallel,
            sf_parameters.n_cpu,
            sf_parameters.custom_scorers,
            sf_parameters.custom_scorer_weights
        )
