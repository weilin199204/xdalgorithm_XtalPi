import os

from .base import TaskBase
from .base import UNDEFINED_PARAMETER

__all__ = [
    'MDSystemBuilder',
]


class MDSystemBuilder(TaskBase):
    def __init__(self, name='md_system_builder'):
        """ MD System Building Task

        Args:
            name (str,optional): the task name. Default to 'md_system_builder'

        Examples:
        >>> from xdalgorithm.engines import get_dataset
        ... from xdalgorithm.engines.md import MDSystemBuilder
        ... from xdalgorithm.engines import SerialEventBase

        >>> data=get_dataset()

        >>> data.run_task(
        ...     SerialEventBase,
        ...     input_node_types=['0.HYPOTHESIS', '2.ligand_conformation_generater'],
        ... )
        """
        super().__init__(name=name)
        self.config_template = {
            'protein_pdb_file_name': UNDEFINED_PARAMETER,
            'ligand_sdf_file_name': UNDEFINED_PARAMETER,
            'ligand_resname': 'MOL',
            'ligand_charge_method': 'bcc',
            'water_box_shape': 'cubic',
            'water_box_edge': 1.0,
            'ff_format': 'AMBER',
            'protein_prmtop_file_name': 'protein.prmtop',
            'protein_inpcrd_file_name': 'protein.inpcrd',
            'ligand_prmtop_file_name': 'ligand.prmtop',
            'ligand_inpcrd_file_name': 'ligand.inpcrd',
            'system_unsolvated_prmtop_file_name': 'system_unsolvated.prmtop',
            'system_unsolvated_inpcrd_file_name': 'system_unsolvated.inpcrd',
            'system_unsolvated_psf_file_name': 'system_unsolvated.psf',
            'system_unsolvated_pdb_file_name': 'system_unsolvated.pdb',
            'system_unsolvated_top_file_name': 'system_unsolvated.top',
            'system_unsolvated_gro_file_name': 'system_unsolvated.gro',
            'system_solvated_prmtop_file_name': 'system_solvated.prmtop',
            'system_solvated_inpcrd_file_name': 'system_solvated.inpcrd',
            'system_solvated_psf_file_name': 'system_solvated.psf',
            'system_solvated_pdb_file_name': 'system_solvated.pdb',
            'system_solvated_top_file_name': 'system_solvated.top',
            'system_solvated_gro_file_name': 'system_solvated.gro',
            'working_dir_name': None
        }

    def run(self):
        from xdalgorithm.toolbox.md.system_building.system_builder_runner import MDSystemBuilderRunner
        md_system_builder_runner = MDSystemBuilderRunner(self.config_template['protein_pdb_file_name'],
                                                         self.config_template['ligand_sdf_file_name'],
                                                         ligand_resname=self.config_template['ligand_resname'],
                                                         ligand_charge_method=self.config_template['ligand_charge_method'],
                                                         water_box_shape=self.config_template['water_box_shape'],
                                                         water_box_edge=self.config_template['water_box_edge'],
                                                         ff_format=self.config_template['ff_format'],
                                                         protein_prmtop_file_name=self.config_template['protein_prmtop_file_name'],
                                                         protein_inpcrd_file_name=self.config_template['protein_inpcrd_file_name'],
                                                         ligand_prmtop_file_name=self.config_template['ligand_prmtop_file_name'],
                                                         ligand_inpcrd_file_name=self.config_template['ligand_inpcrd_file_name'],
                                                         system_unsolvated_prmtop_file_name=self.config_template['system_unsolvated_prmtop_file_name'],
                                                         system_unsolvated_inpcrd_file_name=self.config_template['system_unsolvated_inpcrd_file_name'],
                                                         system_unsolvated_psf_file_name=self.config_template['system_unsolvated_psf_file_name'],
                                                         system_unsolvated_pdb_file_name=self.config_template['system_unsolvated_pdb_file_name'],
                                                         system_unsolvated_top_file_name=self.config_template['system_unsolvated_top_file_name'],
                                                         system_unsolvated_gro_file_name=self.config_template['system_unsolvated_gro_file_name'],
                                                         system_solvated_prmtop_file_name=self.config_template['system_solvated_prmtop_file_name'],
                                                         system_solvated_inpcrd_file_name=self.config_template['system_solvated_inpcrd_file_name'],
                                                         system_solvated_psf_file_name=self.config_template['system_solvated_psf_file_name'],
                                                         system_solvated_pdb_file_name=self.config_template['system_solvated_pdb_file_name'],
                                                         system_solvated_top_file_name=self.config_template['system_solvated_top_file_name'],
                                                         system_solvated_gro_file_name=self.config_template['system_solvated_gro_file_name'],
                                                         working_dir_name=self.config_template['working_dir_name'])

        md_system_file_names_dict = md_system_builder_runner.run()
        return [md_system_file_names_dict]
