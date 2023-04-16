from .base_parameter_helper import BaseParameterHelper

class ShapeAlignParameterHelper(BaseParameterHelper):
    """
    A helper to generate the parameter template of training shape-based scoring function.

    Template:

        ```
        {
            "component_type":"shape",
            "name":"shape",
            "weight":1,
            "model_path":None,
            "smiles":[],
            "specific_parameters":{
                "ref_sdf":"lig.sdf",
                "max_confshapes":50
            }
        }
        ```

    Usage:
        ```
        sdfile = 'lig.sdf'
        numConfs = 50
        shapeHelper = ShapeAlignParameterHelper(
            sdfile,
            numConfs
        )
        shapeHelper.generateTemplate()
        ```

    """
    JSON_TEMPLATE = {
        "component_type":"shape",
        "name":"shape",
        "weight":1,
        "model_path":None,
        "smiles":[],
        "specific_parameters":{
            "ref_sdf":"lig.sdf",
            "max_confshapes":50
        }
    }

    def __init__(self, sdf, numConfs):
        self.sdf = sdf
        self.numConfs = numConfs

    def generate_template(self):
        self.JSON_TEMPLATE['specific_parameters']['ref_sdf'] = self.sdf
        self.JSON_TEMPLATE['specific_parameters']['max_confshapes'] = self.numConfs
        return self.JSON_TEMPLATE