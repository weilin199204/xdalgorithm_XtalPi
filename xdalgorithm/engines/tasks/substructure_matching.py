from rdkit import Chem
from .base import UNDEFINED_PARAMETER
from .base import TaskBase


class Filtering(TaskBase):
    def __init__(self, name="substructure_matching"):
        """perform substructure matching for input SMILES

        Args:
            name (str,optional): the task name. Default to 'substructure_matching'

        Examples:
        >>> from xdalgorithm.engines import get_dataset
        ... from xdalgorithm.engines import SubstructureMatching
        ... from xdalgorithm.engines import SerialEventBase

        >>> data=get_dataset()

        >>> data.run_task(
        ...     SerialEventBase,
        ...     task=SubstructureMatching(),
        ...     input_node_types=['0.LIGAND'],
        ...     name='substructure_matching_1',
        ...     component_type = 'pains'
        ... )
        {'substructure_matching_1_reject': 0, 'substructure_matching_1_accept': 480, 'valid': 1, 'layer': 1}
        """
        super().__init__(name)
        self.config_template = {
            'component_type': UNDEFINED_PARAMETER,
            'name': None,
            'SMILES': None,
        }

    def run(self):
        from xdalgorithm.toolbox.scoring_filtering.substruture_matching_runner import SubstructureMatchingRunner
        runner = SubstructureMatchingRunner(component_type=self.config_template['component_type'],
                                            name=self.config_template['name'],
                                            input_smiles_list=[self.config_template['SMILES']])
        scoring_df = runner.run()
        scoring_df = scoring_df.drop(['smiles'], axis=1)
        node_dict = scoring_df.T.to_dict()[0]
        return [node_dict]
