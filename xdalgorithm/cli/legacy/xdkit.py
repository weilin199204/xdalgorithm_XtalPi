import os


class XDKit(object):
    def __init__(self, config, json_file_name):
        self.json_file_name = json_file_name
        self.config = config
        self.run_type = self.config['run_type']

    def run(self):
        reinvent_cli_list = ['sampling', 'transfer_learning', 'reinforcement_learning', 'create_model']
        mmp_cli_list = ['mmp_grow', 'mmp_link', 'mmp_mutate']
        xreact_cli_list = ['xreact_retrosynthesis', 'xreact_rebuild', 'xreact_clustering']
        screen_cli_list = ['scoring', 'filtering']

        if self.run_type in reinvent_cli_list:
            os.system('xdd molgen -f ' + self.json_file_name)
        elif self.run_type in mmp_cli_list:
            os.system('xdd mmp -f ' + self.json_file_name)
        elif self.run_type in xreact_cli_list:
            os.system('xdd xreact -f ' + self.json_file_name)
        elif self.run_type in screen_cli_list:
            os.system('xdd screen -f ' + self.json_file_name)
        elif self.run_type == 'protein_fixer':
            os.system('xdd pdbprep -f ' + self.json_file_name)
        elif self.run_type == 'ligand_dock':
            os.system('xdd dock -f ' + self.json_file_name)
        elif self.run_type == 'core_constraint_docking':
            os.system('xdd cdock -f ' + self.json_file_name)
        elif self.run_type == 'md_system_building':
            os.system('xdd md_system_building -f ' + self.json_file_name)
        elif self.run_type == 'md_sampling':
            os.system('xdd md_sampling -f ' + self.json_file_name)
            os.system('xdd md_trajectory_analysis -f ' + self.json_file_name)


class CLICommand:
    """An generic entry and wrapper for the xdd command line interface.
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-f', '--input-json', default='', type=str)

    @staticmethod
    def run(args):
        from xdalgorithm.utils import load_arguments_from_json
        configuration = load_arguments_from_json(args.input_json)
        xdkit_runner = XDKit(configuration, args.input_json)
        xdkit_runner.run()
