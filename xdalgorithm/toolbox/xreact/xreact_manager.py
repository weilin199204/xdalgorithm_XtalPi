from xdalgorithm.toolbox.xreact.rebuild import Rebuild
from xdalgorithm.toolbox.xreact.clustering import Clustering
from xdalgorithm.toolbox.xreact.retro_routes_runner import RetrosynthesisRoutesRunner
from xdalgorithm.toolbox.xreact.utils import get_xreact_reaction_path
from xdalgorithm.toolbox.xreact.fragment_connection import FragmentConnection
import os
import json


class XreactManager:
    def __init__(self, configuration):
        self.configuration = configuration
        self.db_path = self.configuration.get("db_path", get_xreact_reaction_path())

    def _run_rebuild(self):
        # load parameters
        route = self.configuration.get('route')
        core_smiles = self.configuration.get('core_smiles')
        output_path = self.configuration.get('output_path')
        bb_dict = self.configuration.get('bb_dict')
        n_cpu = self.configuration.get('n_cpu', 8)

        rb = Rebuild(route=route, db_path=self.db_path, core=core_smiles,
                     output_path=output_path, bb_dict=bb_dict, n_cpu=n_cpu)
        rb.rebuildByRoute()

    def _run_clustering(self):
        reactant_def = self.configuration.get('reactant_def')
        bb_from = self.configuration.get('bb_from')
        rep = self.configuration.get('rep')
        cutoff = self.configuration.get('cutoff')
        n_cpu = self.configuration.get('n_cpu', 8)
        logging_path = self.configuration.get('logging_path', ValueError())

        cl = Clustering(reactant_def, bb_from, rep, cutoff, self.db_path,
                        n_cpu=n_cpu, logging_path=logging_path)
        cl.run()

    def _run_fragment_connection(self):
        bb_from = self.configuration.get('bb_from')
        query_smiles = self.configuration.get('query_smiles')
        output_path = self.configuration.get('output_path')
        n_cpu = self.configuration.get('n_cpu', 8)
        parallel = True
        fragment_connection = FragmentConnection(bb_from=bb_from,
                                                 query_smiles=query_smiles,
                                                 output_path=output_path,
                                                 parallel=parallel,
                                                 n_cpu=n_cpu)
        fragment_connection.run()

    def _run_retrosynthesis(self):
        start_smiles = self.configuration.get('start_smiles', ValueError)
        core_smiles = self.configuration.get('core_smiles', ValueError)
        logging_path = self.configuration.get('logging_path', ValueError)
        core_specific = self.configuration.get('core_specific', True)
        core_single_reactive_center = self.configuration.get('core_single_reactive_center', True)

        # save images firstly
        if not os.path.exists(logging_path):
            os.mkdir(logging_path)

        # create a temporary folder
        temp_directory = os.path.join(logging_path, 'temp_routes_imgs')
        if not os.path.exists(temp_directory):
            os.mkdir(temp_directory)

        retrosynthesis_routes_runner = RetrosynthesisRoutesRunner(start_smiles=start_smiles,
                                                                  core_smiles=core_smiles,
                                                                  logging_path=temp_directory,
                                                                  core_specific=core_specific,
                                                                  core_single_reactive_center=core_single_reactive_center,
                                                                  db_path=self.db_path)
        unique_routes_list, routes_imgs, building_block_path_list = retrosynthesis_routes_runner.run()

        # create a directory to save images
        img_directory = os.path.join(logging_path, 'routes_imgs')
        if not os.path.exists(img_directory):
            os.mkdir(img_directory)

        img_path_list = []
        for i, img in enumerate(routes_imgs):
            img_path = os.path.join(img_directory, 'routes_img_{0}.png'.format(i))
            img.save(img_path)
            img_path_list.append(img_path)
        # json dictionary
        route_dictionary = {'routes': unique_routes_list,
                            'images': img_path_list,
                            'bb_dict_list': building_block_path_list}
        # output a json and a list of images
        routes_json_file = os.path.join(logging_path, 'retrosynthesis_routes.json')
        routes_json_str = json.dumps(route_dictionary, default=lambda x: x.__dict__, sort_keys=True, indent=4,
                                     separators=(',', ': '))
        with open(routes_json_file, 'w') as f:
            f.write(routes_json_str)

    def _save_configuration(self):
        if not os.path.exists(self.configuration.get("logging_path")):
            os.mkdir(self.configuration.get("logging_path"))
        file = os.path.join(self.configuration.get("logging_path"), "input.json")
        jsonstr = json.dumps(self.configuration, default=lambda x: x.__dict__, sort_keys=True, indent=4,
                             separators=(',', ': '))
        with open(file, 'w') as f:
            f.write(jsonstr)

    def run(self):
        if self.configuration.get('run_type') == 'xreact_rebuild':
            self._run_rebuild()
            self._save_configuration()
        if self.configuration.get('run_type') == 'xreact_clustering':
            self._run_clustering()
            self._save_configuration()
        if self.configuration.get('run_type') == 'xreact_retrosynthesis':
            self._run_retrosynthesis()
            self._save_configuration()
        if self.configuration.get('run_type') == "xreact_connection":
            self._run_fragment_connection()
            self._save_configuration()
