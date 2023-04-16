from .base_parameter_helper import BaseParameterHelper

class DiverseFilterParameterHelper(BaseParameterHelper):
    """
    A helper to generate the parameter templates for diverse filter.

    Template:

        ```
        {
            "name": "IdenticalMurckoScaffold",
            "nbmax": 64,
            "minscore": 0.5,
            "minsimilarity": 0.5
        }
        ```

    Usage:
        ```
        divFilterHelper = DiverseFilterParameterHelper(
            nbmax = 64, minscore = 0.5， minsimilarity = 0.5
        )
        divFilterHelper.generate_template()
        ```

    """
    JSON_TEMPLATE = {
        "name": "IdenticalMurckoScaffold",
        "nbmax": 64,
        "minscore": 0.5,
        "minsimilarity": 0.5
    }

    def __init__(self, nbmax, scoreThreshold, similarityThreshold):
        self.nbmax = nbmax
        self.scoreThreshold = scoreThreshold
        self.similarityThreshold = similarityThreshold

    def generate_template(self):
        self.JSON_TEMPLATE['nbmax'] = 64
        self.JSON_TEMPLATE['minscore'] = 0.5
        self.JSON_TEMPLATE['minsimilarity'] = 0.5
        return self.JSON_TEMPLATE