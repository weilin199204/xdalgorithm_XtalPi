"""
The best practice is to use `get_dataset` function.
"""
import typing as t
from itertools import (
    product,
    chain,
    combinations
)
from collections import defaultdict
from copy import deepcopy
import os
import time
import dill
import traceback
import warnings
import re

from eventsourcing.domain.model.aggregate import BaseAggregateRoot
import networkx as nx
from networkx.algorithms.dag import (
    ancestors,
    descendants
)
from networkx.algorithms.simple_paths import all_simple_paths
from networkx.algorithms.shortest_paths.generic import has_path

# from xdalgorithm.utils import get_rand_id
from xdalgorithm.engines.utils import (
    level_changed,
)
from .tasks.merge_nodes import Merge
from .events import CollectiveEventBase
from .utils import (
    draw_graph,
    DefaultOrderedDict
)

__all__ = [
    'get_dataset',
]


warnings.filterwarnings('ignore', message='.*CRYST1.*')
warnings.filterwarnings('ignore', message='.*Unit cell dimensions not found.*')


def get_dataset(
    working_dir_name: str = ".",
    **kwargs
) -> 'Dataset':
    """Generate a dataset `AggregateRoot` instance to hold an event-sourcing
    dataset
    
    Args:
        working_dir_name (str): working directory.

    Returns:
        Dataset: an `AggregateRoot` `Dataset` instance
    
    Examples:
    >>> data = get_dataset()
    >>> data.run_task(
    ...     Event,
    ...     task=Task(),
    ...     **event_kwargs,
    ... )
    
    >>> # load local graph_file
    >>> from xdalgorithm.engines import load_data, ch_datadir
    >>> load_data(data, 5) 
    """
    print("Creating your dataset manager...")
    data = Dataset.__create__(**kwargs)
    if working_dir_name.endswith('/'):
        working_dir_name = working_dir_name.rstrip('/')
    data.working_dir_name = os.path.abspath(working_dir_name)
    os.makedirs(data.working_dir_name, exist_ok=True)
    print(f"Completed! Your working directory is: {data.working_dir_name}")
    # data.snapshot()
    return data


class Dataset(BaseAggregateRoot):
    _SPECIAL_TYPES = ('MODEL', 'HYPOTHESIS')

    def __init__(
        self,
        **kwargs
    ):
        super(Dataset, self).__init__(**kwargs)

        """
        node_type_dict: {node_type: list of nodes, ...}
        node_types: 
            - 'layer_id.HYPOTHESIS.hypothesis_name'
            - 'layer_id.MODEL.model_name'
            - 'layer_id.LIGAND.task_name'

        event_types: 'MODEL', 'HYPOTHESIS', or 'LIGAND'
        """

        self.node_type_dict = defaultdict(list)
        self.node_max_id = 0
        self.layer_id = 0

        self.graph = nx.DiGraph()
        self._working_dir_name = None
        self.snapshot_idx = -1
        
        self.task_dirs = []

        self.level_dict = {
            'LIGAND': 2,
            'MODEL': 1,
            'HYPOTHESIS': 0
        }

        # self.connected_node_pairs = set([])
        self.connected_nodes_dict = DefaultOrderedDict(set)
    
    @property
    def working_dir_name(self):
        return self._working_dir_name
    
    def get_node_level(self, node: int):
        return self.level_dict[self.graph.nodes[node]['type']]
    
    def get_type_level(self, node_type: str):
        return self.level_dict[node_type.split('.')[1]]
    
    @working_dir_name.setter
    def working_dir_name(self, working_dir):
        if working_dir.endswith('/'):
            working_dir = working_dir.rstrip('/')
        self._working_dir_name = os.path.abspath(working_dir)
    
    def _node_rep(self, node) -> str:
        return (
            f"{self.graph.nodes[node]['layer']}."
            + f"{self.graph.nodes[node]['type']}."
            + f"{self.graph.nodes[node]['name']}"
        )

    def _all_predecessors(self, node, overview) -> t.Set:
        """get all precessor nodes of a node in self.graph

        Args:
            node (int): a node in self.graph

        Returns:
            t.Set: all tracable precessor nodes of the node
        """
               
        def _cannot_trace(i, j, overview) -> bool:
            if overview:
                level_func = self.get_type_level
            else:
                level_func = self.get_node_level

            if level_func(i) >= level_func(j):
                return False
            else:
                return True
        
        _precessors = set([node])

        if overview:
            graph = self.overview()
        else:
            graph = self.graph

        _p = [
            x
            for x in graph.predecessors(node)
            # if not _cannot_trace(node, x, overview)
        ]
        # print(_p)

        while _p:
            precessor_node = _p.pop()
            _precessors.add(precessor_node)
            _p += [
                x
                for x in graph.predecessors(precessor_node)
                # if self.get_node_level(x) <= self.get_node_level(precessor_node)
                # if not _cannot_trace(precessor_node, x, overview)
            ]
        
        return _precessors 

    def snapshot(self):
        assert self.working_dir_name is not None, \
            "You should use get_dataset() to create a Dataset"
        print("\nDumping new dataset snapshot...")
        self.snapshot_idx += 1
        nx.write_gpickle(
            self.graph,
            self.working_dir_name + '/' + f'graph_{self.snapshot_idx}.gpickle'
        ) 
        state_dict = {}
        state_dict['node_max_id'] = self.node_max_id
        state_dict['node_type_dict'] = deepcopy(self.node_type_dict)
        state_dict['snapshot_idx'] = self.snapshot_idx
        state_dict['task_dirs'] = self.task_dirs
        state_dict['layer_id'] = self.layer_id
        state_dict['connected_nodes_dict'] = self.connected_nodes_dict
        with open(
            self.working_dir_name + '/' + f'state_{self.snapshot_idx}.pkl', 'wb'
        ) as f:
            dill.dump(state_dict, f)
        print(f'NEW snapshot! Index: {self.snapshot_idx}')

    def overview(self):  # called every time generating task config
        """
        Returns:
            a graph with connected node types
        """
        g = nx.DiGraph()
        for _t in self.node_type_dict:  # _t: str
            if not g.has_node(_t):
                g.add_node(_t)

            _n = self.node_type_dict[_t][0]  # use the first node in self.graph
            for _p in self.graph.predecessors(_n):
                _node_name = self._node_rep(_p)

                if not g.has_node(_node_name):
                    g.add_node(_node_name)
                
                g.add_edge(_node_name, _t)
        return g
    
    def induced_og(self, event_type, node_types):

        og = self.overview()
        new_g = nx.DiGraph()
        nodes = set(node_types)
        new_g.add_nodes_from(nodes)
        terminals, seen_nodes = set(), set()

        for i, j in combinations(nodes, 2):
            path_flag = False
            if i in ancestors(og, j) or i in descendants(og, j):
                if i in ancestors(og, j):
                    source, target = i, j
                else:
                    source, target = j, i

                if og.has_edge(source, target):
                    new_g.add_edge(source, target)
                    path_flag = True
                else:
                    paths = list(all_simple_paths(og, source, target))
                    for path in paths:
                        if len(set(path) & set(node_types)) == 2:
                        
                            levels = [
                                self.get_type_level(node_type)
                                for node_type in path
                            ]

                            if not level_changed(levels):
                                new_g.add_edge(source, target)
                                path_flag = True
                                break
                
                if path_flag:
                    if self.get_type_level(source) > self.get_type_level(target):
                        if not source in seen_nodes and \
                            self.level_dict[event_type] >= self.get_type_level(source):
                            terminals.add(source)

                    elif source in terminals:
                        terminals.remove(source)

                    if not target in seen_nodes:
                        terminals.add(target)
                        
                    seen_nodes.update([source, target])

            elif any(ancestors(og, i) & ancestors(og, j)):
                common_ancestors = ancestors(og, i) & ancestors(og, j)
                # num_ancestors_in_newg = 0
                for common_ancestor in common_ancestors:
                    # print(i, j, common_ancestor)
                    paths_i, paths_j = (
                        all_simple_paths(og, common_ancestor, i),
                        all_simple_paths(og, common_ancestor, j)
                    )
                    
                    check_ancestor_i, check_ancestor_j = False, False
                    for path_i in paths_i:
                        levels = [
                            self.get_type_level(node_type)
                            for node_type in path_i
                        ]
                        path_i_throughs = deepcopy(path_i)
                        path_i_throughs.remove(common_ancestor)
                        path_i_throughs.remove(i)

                        if (
                            (not level_changed(levels))
                            and (len(set(path_i_throughs) & set(node_types)) == 0)
                        ):
                            check_ancestor_i = True
                            break
                    
                    for path_j in paths_j:
                        levels = [
                            self.get_type_level(node_type)
                            for node_type in path_j
                        ]
                        path_j_throughs = deepcopy(path_j)
                        path_j_throughs.remove(common_ancestor)
                        path_j_throughs.remove(j)

                        if (
                            (not level_changed(levels))
                            and (len(set(path_j_throughs) & set(node_types)) == 0)
                        ):
                            check_ancestor_j = True
                            break
                    
                    if check_ancestor_i and check_ancestor_j:
                        if common_ancestor in node_types:
                            new_g.add_edge(common_ancestor, i)
                            new_g.add_edge(common_ancestor, j)
                        else:
                            new_g.add_edge(i, j)

                        if i not in seen_nodes:
                            terminals.add(i)
                        if j not in seen_nodes:
                            terminals.add(j)
                        # seen_nodes.update([i, j])

        new_g = new_g.to_undirected()

        return new_g, terminals
    
    def update_result(
        self,
        result: t.Dict,
        event_type: str,
        event_name: str,
        source: t.List[int],
    ):

        """Update graph of Dataset
        Record results as nodes
        Connect source input nodes with the new nodes
        This function add ONE node each time (each dict in the results list) 
            and connect every node in `source` 
            to this node.

        Args:
            results (t.List[t.Dict]):
                results from triggered events.
            event_type (str):
                the event_type, 'MODEL', 'HYPOTHESIS', or 'LIGAND'. 
            event_name (str): 
                the event_name, task name defined in the TaskBase
            source (t.List[int]): 
                the node_ids to be connected, already in self.graph.  
        """

        target = self.node_max_id

        result['layer'] = self.layer_id
        result['type'] = event_type
        result['name'] = event_name
        type_name = f"{self.layer_id}.{event_type}.{event_name}"
        self.node_type_dict[type_name].append(target)
        self.graph.add_node(target, **result)

        for s in source:
            self.graph.add_edge(s, target)

            # for _key in self.connected_nodes_dict:
            #     if s in self.connected_nodes_dict[_key]:
            #         self.connected_nodes_dict[_key].add(target)
                
            # self.connected_nodes_dict[s].add(target)
            self.connected_nodes_dict[target].add(s)
            if s in self.connected_nodes_dict:
                self.connected_nodes_dict[target] = set.union(
                    self.connected_nodes_dict[target],
                    self.connected_nodes_dict[s]
                )
            
            # tmp_connected_pairs = deepcopy(self.connected_node_pairs)
            # for node_pair in tmp_connected_pairs:
            #     if s == node_pair[1]:
            #         self.connected_node_pairs.add((node_pair[0], target))
            # del tmp_connected_pairs
            # self.connected_node_pairs.add((s, target))
            # # gc.collect()

        self.node_max_id += 1

    def query_nodes(
        self,
        node_type: str,
        check_keys: t.List[str] = [],
    ) -> t.List[int]:
        """query nodes by task and key
        * For updatable hypothsis node, only return the latest node.

        Args:
            node_type (str): 
                event type, either in self._SPECIAL_TYPES or layer_id.task_name
            check_keys (list of str, optional, default to []): 
                make sure the keys exist

        Returns:
            t.List[int]: quried nodes ids.
        """
        nodes = []
        if node_type not in self.node_type_dict:
            print(
                'WARNING: Cannot find node type %s in the dataset!' % node_type
            )
        else: 
            for n in self.node_type_dict[node_type]:
                if check_keys:
                    for k in check_keys:
                        if k not in self.graph.nodes[n]:
                            print(f"ERROR: {k} does not exist!")
                            return []
                nodes.append(n)
        return nodes

    def _generate_task_config(
        self,
        continue_job,
        event_kwargs,
    ) -> t.List[t.Dict]:
        """
            get input nodes by querying input_node_types
            return a list of configurations for events
        """
        # task_name = event_kwargs['task'].name
        # event type is `LIGAND` by default
        
        source_types = []  # len=num(dependency chains)  list of list of str
        if 'event_type' not in event_kwargs:
            event_kwargs['event_type'] = 'LIGAND'

        event_type = event_kwargs['event_type']
        event_name = event_kwargs['task'].name
        current_type = f'{self.layer_id}.{event_type}.{event_name}'
        og = self.overview()
        if any(self.node_type_dict.keys()):
            print("The whole overview graph")
            draw_graph(og)
        if not continue_job and og.has_node(current_type):
            print('Existed task. Exit!')
            return []

        # source nodes are obtained from the end nodes of the dependency chain
        independent_nt = []  # t.List[str]: node types to be fully connected
        other_nt = []  # t.List[str] node types in dependency chains
        # collect inputs for task arguments
        inputs = defaultdict(list)
        # inputs = {
        #     input_node_type: [keyword1, keyword2, ...], 
        #     ...
        # }
        variable_args_dict = {}  # {str(key_word:node_type): kw_name}
        fix_args_dict = {}  # {kw_name: kw_value} user defined args

        # print("Collecting inputs args")
        for k, v in event_kwargs.items():  # k: kw_name, v: kw_value
            # if the args should be queried from self.graph
            if isinstance(v, str) and v.startswith('i:'):  # i:keyword:node_type
                keyword, node_type = v.split(':')[1:3]
                inputs[node_type].append(keyword)
                assert node_type in self.node_type_dict.keys()

                if len(v.split(':')) == 4 and v.split(':')[-1] == 'u':
                    independent_nt.append(node_type)
                else:
                    other_nt.append(node_type)

                # only for input from self.graph
                variable_args_dict[f"{keyword}:{node_type}"] = k  # {str(key_word:node_type): kw_name}
            else:
                # other user defined params
                fix_args_dict[k] = v  # {kw_name: kw_value}

        # no need to connect, just add nodes to self.graph 
        if not inputs:  # do not query self.graph
            event_kwargs['source'] = [] 
            event_kwargs['task'].update_config(**event_kwargs)
            # only on task config
            return [event_kwargs]  # [{'task': TaskBase, other_configs...}]

        # get the graph for the task
        print("Checking dependencies")

        # print(other_nt)
        induced_og, terminals = self.induced_og(event_type, other_nt)
        draw_graph(induced_og) 
        # for edge in induced_og.edges():
        #     print(edge)

        all_node_lists = []  # len = num(dependency chains)  list of list of list of int
        nt_lists = []  # len = num(dependency chains)  list of list of str
        # print(type(induced_og))
        num_chains = 0
        for component in nx.connected_components(induced_og):
            # print(component)
            # print(type(component))
            if len(component) == 1:
                independent_nt.append(component.pop())
            else:
                num_chains += 1
                # max_i = 0  # each sub graph
                component = list(component)
                all_node_lists.append([])
                nt_lists.append([])
                for nt in component:
                    all_node_lists[-1].append(
                        self.query_nodes(
                            node_type=nt, check_keys=inputs[nt]
                        )
                    )
                    nt_lists[-1].append(nt)
                    
                source_types.append([x for x in component if x in terminals]) 
            
        _configs = [[fix_args_dict]]
        _source_nodes = []  # collect node id

        print(f"    Num of dependency chains: {str(num_chains)}\n")

        # collect inputs from ``grouped_node_types``
        print("Collecting configs in dependencies ...")
        # print(nt_lists)
        # print(all_node_lists)

        # print(source_types)
        num_dependency_tasks = 0
        for chain_idx, (nodes, ntl, st_list) in enumerate(
            zip(all_node_lists, nt_lists, source_types)
        ):
            # print(nodes)
            # print(ntl)
            # print(st_list)
            # nodes: list of list of nodes
            # ntl: node type list
            # st_list: node types to be connected
            print(f'    Collecting chain: {str(chain_idx)}')
            print(f'      Node types:')
            for node_type in ntl:
                print(f'        {node_type}')
            _tmp_configs = []
            _source_nodes.append([])                

            list_of_node_list = list(product(*nodes))  # node combinations

            # ttt = 0
            for node_list in list_of_node_list:

                # print(node_list)
                # _tmp_sources = []
                todo = True
                # filter out the unconnected node combinations
                for e in nx.edges(induced_og):
                    # print(e)
                    if not (e[0] in ntl and e[1] in ntl):
                        continue
                    # print(e[0], e[1])
                    i = ntl.index(e[0])
                    j = ntl.index(e[1])
                    # print(node_list[i], node_list[j])
                    node_i = node_list[i]
                    node_j = node_list[j]

                    if (
                        has_path(og, e[0], e[1])
                        or has_path(og, e[1], e[0])
                    ):
                        if (
                            node_i in self.connected_nodes_dict[node_j]
                            or node_j in self.connected_nodes_dict[node_i]
                        ):
                            has_common_predecesors = True
                        else:
                            has_common_predecesors = False

                    else:
                        common_predecessors = (
                            self._all_predecessors(node_i, False)
                            & self._all_predecessors(node_j, False)
                        )
                        has_common_predecesors = len(common_predecessors) > 0
                        # has_common_predecesors = self.has_common_predecesors(
                        #     node_i, node_j
                        # )

                    if not has_common_predecesors:
                        todo = False
                        break
 
                # print(todo)
                if todo:  # todo == the nodes in the list are connected
                    # print('lala')
                    # if 591 in node_list and node_list[1] not in descendants(self.graph, 591):
                    # 为啥要打印591呢，因为591那天就出现在了我眼前
                    #     print(node_list)
                    num_dependency_tasks += 1
                    # print('yes')
                    _source_nodes[-1].append([])                
                    filled_variable_args = {}
                    # results only connect to the source node in the last layer
                    # _source_nodes = [[int],[int], ...]
                    for nt in st_list:
                        # print(nt)
                        # print(st_list)
                        _source_nodes[-1][-1].append(node_list[ntl.index(nt)])
                    # print(_source_nodes)

                    for k in variable_args_dict:
                        keyword_in_k, nt_in_k = k.split(':')
                        if nt_in_k in ntl:  # ntl is one node type of dependencies
                            filled_variable_args[
                                variable_args_dict[k]
                            ] = self.graph.nodes[
                                node_list[ntl.index(nt_in_k)]
                            ][keyword_in_k]
                            # filled_variable_args[kw_name] == node info in node dict

                    _tmp_configs.append(filled_variable_args)
            _configs.append(_tmp_configs)
        # num_dependency_chains = len(_configs) - 1
        # print(f"Num of dependency chains: {num_dependency_chains}")
        print(f"    Num of tasks of dependencies: {str(num_dependency_tasks)}")
        # print(_source_nodes)

        independent_nt = list(set(independent_nt))

        print("\nCollecting configs for independent nodes")
        for nt in independent_nt:  # not in dependencies
            # print(nt)
            nodes = self.query_nodes(node_type=nt, check_keys=inputs[nt])
            _source_nodes.append(nodes)
            _configs.append([])
        
            for n in nodes:
                filled_variable_args = {}
                for k in variable_args_dict:
                    keyword_in_k, nt_in_k = k.split(':')
                    if nt_in_k == nt:
                        filled_variable_args[variable_args_dict[k]] = \
                            self.graph.nodes[n][keyword_in_k]
                
                _configs[-1].append(filled_variable_args)

        def skip_this_task(_sn, out_event_type, out_event_name):
            output_list = [set(self.graph.neighbors(n)) for n in _sn]
            # print(output_list)
            outputs = set.intersection(*output_list) if any(output_list) else []
            for out_node in outputs:
                if f'{out_event_type}.{out_event_name}' in self._node_rep(out_node):
                    return True

            return False

        # node combinations
        # print(_configs)
        # for i in _configs:
        #     print(i)
        # print(_source_nodes)
        all_configs = list(product(*_configs))
        all_source_nodes = list(product(*_source_nodes))
        # print(all_source_nodes)

        assert len(all_configs) == len(all_source_nodes)
        
        print(f"\nNum of total configs: {len(all_configs)}") 
        todo_tasks = []
        # check if child nodes exist
        print("Excluding existing task configs...")
        # print(all_configs)
        for _config, _source_nodes in zip(all_configs, all_source_nodes): 
            # print(_config)
            _sn = []
            for source_node_group in _source_nodes:
                if isinstance(source_node_group, int):
                    _sn.append(source_node_group)
                elif isinstance(source_node_group, t.Iterable):
                    for node_idx in source_node_group:
                        _sn.append(node_idx)
            _main_config = deepcopy(_config[0])
            for _c in _config:
                _main_config.update(_c)
           
            if not continue_job or (
                continue_job
                and not skip_this_task(_sn, event_type, event_name)
            ):
                task = {}
                task['task'] = deepcopy(event_kwargs['task'])
                task['task'].update_config(**_main_config)

                task['source'] = list(set(_sn))
                task['event_type'] = event_kwargs['event_type']
                todo_tasks.append(task)

        print(f"Num of todo tasks: {len(todo_tasks)}")
        return todo_tasks

    def run_task(
        self,
        event: BaseAggregateRoot.Event,
        continue_job=False,
        **event_kwargs
    ):
        """
        continue_job, boolean

        keywords in event_kwargs:
        - task: Task instance
        - args: Task arguments 
        - if format as 'i:keyword:node_type(:u)', query Dataset.graph for inputs
        - else defined by user inputs

        - event_type: 
            - HYPOTHESIS: define the hypothesis
            - MODEL: define the model
            - LIGAND: default
        """

        assert self.working_dir_name, \
            "You should use get_dataset() to create a Dataset"
        # len(intput_node_types) == len(query_key)

        assert 'task' in event_kwargs
        task_name = event_kwargs['task'].name
        task_time = time.strftime('%Y%m%d_%H%M%S')
        task_work_dir = os.path.join(
            self.working_dir_name,
            f"{task_name}_{task_time}"
        )
        print(f"The task directory is: {task_work_dir}\n")
        self.task_dirs.append(task_work_dir)
        os.makedirs(task_work_dir, exist_ok=True)
        event_kwargs['working_dir_name'] = task_work_dir
        # event_kwargs['task'].update_config(**event_kwargs)

        _configs = self._generate_task_config(continue_job, event_kwargs)

        # print(_configs)

        try:
            if len(_configs) > 0:
                self.__trigger_event__(event, what=_configs)
            else:
                print('Nothing to do. Exit!')
        except Exception as e:
            print(e)
            traceback.print_tb(e.__traceback__)

    def merge_nodes(
        self,
        node_types: t.List[str],
        **event_kwargs,
    ):
        """Merge nodes into a same node type

        Args:
            node_types (t.List[str]): the node types with same configs
        
        Examples:
        >>> data.merge_nodes(['5.LIGAND.autodock', '6.LIGAND.autodock'])
        """
        nodes = [
            self.query_nodes(node_type=nt, check_keys=[])
            for nt in node_types 
        ]

        nodes = list(chain.from_iterable(nodes))
        key_template = self.graph.nodes[nodes[0]].keys()
        # layer_max = self.graph.nodes[nodes[0]]['layer']
        for node_idx in range(1, len(nodes)):
            assert self.graph.nodes[nodes[node_idx]].keys() == key_template, \
                "Nodes to be merged should have the same keys!"

        event_kwargs = []
        # event_kwargs['task'] = []
        merge_names = [node_type.replace('.', '_') for node_type in node_types]

        for node in nodes:
            _config = {}
            merge_task = Merge(
                # name='merge_' + '_'.join([x.split('.')[1] for x in node_types])
                name='merge_' + '_'.join(merge_names)
            )
            merge_task.config_template.update(self.graph.nodes[node])
            # merge_task.config_template['layer'] = layer_max 
            _config['task'] = merge_task
            _config['event_type'] = 'MERGE'
            _config['source'] = [node]
            event_kwargs.append(_config)

        try:
            self.__trigger_event__(CollectiveEventBase, what=event_kwargs)
        except Exception as e:
            print(e)
            traceback.print_tb(e.__traceback__)

        print(f"Merge done! New node type: {'merge_' + '_'.join(merge_names)}")
