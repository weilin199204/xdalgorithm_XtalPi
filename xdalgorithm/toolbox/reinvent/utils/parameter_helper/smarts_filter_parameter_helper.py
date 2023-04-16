from .base_parameter_helper import BaseParameterHelper

class SmartsFilterParameterHelper(BaseParameterHelper):
    """
    A helper to generate the template of training smarts-based scoring function.

    Template:

        ```
        {
            "component_type": "custom_alerts",
            "name": "remove_old_scaffold",
            "weight": 1,
            "model_path": None,
            "smiles": [
                "[#7,#6;R]1:&@[#7,#6;R]:&@[#7,#6;R]:&@[#7,#6;R](:&@[#7,#6;R](:&@[#7,#6;R]:&@1)-&!@[#7,#6;R]1:&@[#7,#6;R]:&@[#7,#6;R]2:&@[#7,#6;R](:&@[#7,#6;R]:&@1-&!@[#7,#6;R]1:&@[#7,#6;R]:&@[#7,#6;R]:&@[#7,#6;R]:&@[#7,#6;R]:&@[#7,#6;R]:&@1):&@[#7,#6;R]:&@[#7,#6;R]:&@[#6,#7;R]:&@[#7,#6;R]:&@2)"
            ]
        }
        ```
    
    Usage:

        ```
        weight = 1
        modelPath = None
        smartsFilter = [
            "[#7,#6;R]1:&@[#7,#6;R]:&@[#7,#6;R]:&@[#7,#6;R](:&@[#7,#6;R](:&@[#7,#6;R]:&@1)-&!@[#7,#6;R]1:&@[#7,#6;R]:&@[#7,#6;R]2:&@[#7,#6;R](:&@[#7,#6;R]:&@1-&!@[#7,#6;R]1:&@[#7,#6;R]:&@[#7,#6;R]:&@[#7,#6;R]:&@[#7,#6;R]:&@[#7,#6;R]:&@1):&@[#7,#6;R]:&@[#7,#6;R]:&@[#6,#7;R]:&@[#7,#6;R]:&@2)"
        ]
        smartsHelper = SmartsFilterParameterHelper(
            weight,
            modelPath,
            smartsFilter
        )
        smartsHelper.gnerateTemplate()
        ```

    """
    JSON_TEMPLATE = {
        "component_type": "custom_alerts",
        "name": "remove_old_scaffold",
        "weight": 1,
        "model_path": None,
        "smiles": [
            "[#7,#6;R]1:&@[#7,#6;R]:&@[#7,#6;R]:&@[#7,#6;R](:&@[#7,#6;R](:&@[#7,#6;R]:&@1)-&!@[#7,#6;R]1:&@[#7,#6;R]:&@[#7,#6;R]2:&@[#7,#6;R](:&@[#7,#6;R]:&@1-&!@[#7,#6;R]1:&@[#7,#6;R]:&@[#7,#6;R]:&@[#7,#6;R]:&@[#7,#6;R]:&@[#7,#6;R]:&@1):&@[#7,#6;R]:&@[#7,#6;R]:&@[#6,#7;R]:&@[#7,#6;R]:&@2)"
        ]
    }

    def __init__(self, weight, modelPath, smartsFilter):
        self.weight = weight
        self.modelPath = modelPath
        self.smartsFilter = smartsFilter
    
    def generate_tempatle(self):
        self.JSON_TEMPLATE['weight'] = self.weight
        self.JSON_TEMPLATE['model_path'] = self.modelPath
        self.JSON_TEMPLATE['smiles'] = self.smartsFilter
        return self.JSON_TEMPLATE
