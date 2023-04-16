import os
from xdalgorithm.toolbox.xreact.react_bot import ReactBot
from xdalgorithm.toolbox.xreact.unique_routes_parser import Parser


class RetrosynthesisRoutesRunner(object):
    def __init__(self, start_smiles, core_smiles, logging_path, db_path, core_specific=True,
                 core_single_reactive_center=True):
        self._start_smiles = start_smiles
        self._core_smiles = core_smiles
        self._logging_path = logging_path
        self._db_path = db_path
        self._core_specific = core_specific
        self._core_single_reactive_center = core_single_reactive_center

    def run(self):
        react_bot = ReactBot(start_smiles=self._start_smiles,
                             core_smiles=self._core_smiles,
                             db_path=self._db_path)
        react_bot.analysis(core_specific=self._core_specific,
                           core_single_reactive_center=self._core_single_reactive_center)
        unique_routes_list = react_bot.unique_routes
        # get images
        routes_imgs = self._get_imgs_from_routes(unique_routes_list)
        # get building blocks
        building_block_path_list = []
        for each_route in unique_routes_list:
            bb_names = react_bot._runRoute(each_route)
            # save as a dict
            bb_names_dict = dict()
            for bb_name in bb_names:
                full_bb_path = os.path.join(self._db_path, bb_name)
                bb_names_dict[bb_name] = full_bb_path
            building_block_path_list.append(bb_names_dict)
        return unique_routes_list, routes_imgs, building_block_path_list

    def _get_imgs_from_routes(self, unique_routes_list):
        self.temp_dot_path = os.path.join(self._logging_path, 'temp_routes_imgs')
        parser = Parser(unique_routes_list)
        _ = parser.run(self.temp_dot_path)
        return parser.imgs
