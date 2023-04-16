import os

class CLICommand:
    __doc__ = "Auto-analyze and visualize the chemical space of one or several related patents. \n"
    
    _scaffnet_config_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        '..',
        'data',
        'template',
        'scaffold_network',
        'scaffold_network.json'
    )
    with open(_scaffnet_config_path) as f:
        json_raw = f.read()
    
    __doc__ += "Template json file: \n"
    __doc__ += json_raw

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('-f','--input-json',default='',type=str)

    @staticmethod
    def run(args):
        from xdalgorithm.utils import load_arguments_from_json
        from xdalgorithm.toolbox.scaffold_network.scaffold_actions_manager import ScaffoldActionsManager
        configuration = load_arguments_from_json(args.input_json)

        scaffold_network_manager = ScaffoldActionsManager(configuration)
        scaffold_network_manager.run()
