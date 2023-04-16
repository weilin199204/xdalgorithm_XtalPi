# import os
import typing as t

from .base import UNDEFINED_PARAMETER
from .base import TaskBase
from .base import CollectiveTaskBase
from ..utils import (
    merge_as_list,
    merge_dicts_withref_in_list,
    # HashableDict
)

__all__ = [
    "DockingReports",
    "ScoreComp",
]


class DockingReports(CollectiveTaskBase):
    def __init__(self, name: str = 'docking_reports'):
        """[summary]

        Args:
            name (str, optional): [description]. Defaults to 'docking_reports'.
        
        Examples:
        >>> data.run_task(
        ...     CollectiveEventBase,
        ...     task=DockingReports(),
        ...     ligand_molecule_name='i:ligand_molecule_name:0.TASK.add_ligand',
        ...     docking_score='i:docking_score:3.TASK.autodock',
        ...     ligand_efficiency='input:ligand_efficiency:3.TASK.autodock',
        ... )
        """
        super().__init__(name)
        self.config_template = {
            'ligand_molecule_name': UNDEFINED_PARAMETER,
            'docking_score': UNDEFINED_PARAMETER,
            'ligand_efficiency': UNDEFINED_PARAMETER,
        }
    
    def collect_config(self, task: "TaskBase"):
        if isinstance(self.config_template, t.Dict):
            self.config_template = [self.config_template]
        self.config_template.append(task.config_template)

    def run(self):
        result_dict = {}
        result_dict['docking_reports'] = {}

        merged_dict = merge_dicts_withref_in_list(
            self.config_template,
            ['ligand_molecule_name'],
            ['docking_score', 'ligand_efficiency'],
            merge_as_list
        )

        for mol_name_dict, value_dict in merged_dict.items():
            mol_name = mol_name_dict['ligand_molecule_name']
            result_dict['docking_reports'][mol_name] ={}
            result_dict['docking_reports'][mol_name]['docking_score'] = \
                sum(value_dict['docking_score']) / len(value_dict['docking_score'])
            result_dict['docking_reports'][mol_name]['ligand_efficiency'] = \
                sum(value_dict['ligand_efficiency']) / len(value_dict['ligand_efficiency'])

        return [result_dict]
    
        # result_dict = {
        #     'docking_reports': {
        #         'mol_name_1': {
        #             'docking_score': 0.6,
        #             'ligand_efficiency': -0.3272973,
        #         },
        #         'mol_name_2': {
        #             'docking_score': 0.6,
        #             'ligand_efficiency': -0.3272973,
        #         },
        #         ...
        #     }
        # }


class ScoreComp(TaskBase):
    def __init__(self, name: str = 'compare_scores'):
        super().__init__(name=name)
        self.config_template = {
            'predicted_reports': UNDEFINED_PARAMETER,
            'docking_reports': UNDEFINED_PARAMETER
        }
    
    def run(self):
        predicted_reports = self.config_template['predicted_reports']
        docking_reports = self.config_template['docking_reports']
        return [{
            'predicted_reports': predicted_reports,
            'docking_reports': docking_reports
        }]
        