"""
Utility functions. Some generic tools for data processing or visualization
arrows  ↑ ↓ ← → ↖ ↗ ↙ ↘
"""
import typing as t
from collections import OrderedDict 
import numbers
import itertools
from collections import defaultdict
import os
from distutils.dir_util import copy_tree
import dill
import re

import ipycytoscape
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import pylab
from networkx.algorithms.dag import (
    ancestors,
    # descendants
)
# from pyvis import network as net
# from pyvis.network import Network
# from networkx.algorithms.dag import (
#     # ancestors,
#     # descendants
# )
# from networkx.algorithms.simple_paths import all_simple_paths
# import numba

__all__ = [
    "viz_dataset",
    "collect_results",
    "is_descendant",
    "is_descendants_connected_list",
    "share_same_children",
    "get_each_pair_of_list",
    "merge_dicts_withref_in_list",
    "HashableDict",
    "merge_dicts_aslist_withref_in_list",
    'draw_graph',
    'load_data',
    'ch_datadir',
]


def load_data(
    dataset,
    snapshot_idx
):  
    print(f'Loading dataset from snapshot idx {snapshot_idx}')
    gpickle_file = dataset.working_dir_name + '/' + f'graph_{snapshot_idx}.gpickle'
    graph = nx.read_gpickle(gpickle_file)
    dataset.graph = graph
    state_pickle = dataset.working_dir_name + '/' + f'state_{snapshot_idx}.pkl'
    with open(state_pickle, 'rb') as f:
        state_dict = dill.load(f)
    dataset.node_max_id = state_dict['node_max_id']
    # dataset.result_history = state_dict['result_history'] 
    dataset.node_type_dict = state_dict['node_type_dict'] 
    dataset.snapshot_idx = state_dict['snapshot_idx'] 
    # dataset.num_events = state_dict['num_events']
    # dataset.node_eventidx_dict = state_dict['node_eventidx_dict']
    # dataset.lable_eventidx_dict = state_dict['lable_eventidx_dict']
    dataset.task_dirs = state_dict['task_dirs']
    dataset.layer_id = state_dict['layer_id']
    dataset.connected_nodes_dict = state_dict['connected_nodes_dict']
    print('Completed!')


class DefaultOrderedDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
           not isinstance(default_factory, t.Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items()))

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                               OrderedDict.__repr__(self))

def ch_datadir(
    dataset,
    work_dir: str,
    cp_files: bool = False,
    flush: bool = False
):
    """Change dataset workint directory.

    Args:
        dataset (Dataset): a `Dataset`.
        work_dir (str): the new working dir, you can use a relative path.
        cp_files (bool, optional): 
            whether to copy files to new path, only copy files in node infos.
            Defaults to False.
        flush (bool, optional):
            whether to flush node infos using new working dir.
            Defaults to False.
    """
    # if flush:
    #     cp_files = True
    
    old_work_dir = dataset.working_dir_name
    if old_work_dir.endswith('/'):
        old_work_dir = old_work_dir.rstrip('/')
    new_work_dir = os.path.abspath(work_dir)
    if new_work_dir.endswith('/'):
        new_work_dir = new_work_dir.rstrip('/')
    dataset.working_dir_name = new_work_dir
    
    if not os.path.isdir(new_work_dir):
        os.makedirs(new_work_dir, exist_ok=True)

    if flush:
        for node in dataset.graph.nodes():
            for _key, _value in dataset.graph.nodes[node].items():
                if isinstance(_value, str) and _value.startswith(old_work_dir):
                    new_file_path = re.sub(
                        r'^{0}'.format(re.escape(old_work_dir)),
                        new_work_dir, 
                        _value
                    )
                    dataset.graph.nodes[node][_key] = new_file_path
        for i, task_dir in enumerate(dataset.task_dirs):
            new_task_dir = re.sub(
                r'^{0}'.format(re.escape(old_work_dir)),
                new_work_dir, 
                task_dir
            )
            dataset.task_dirs[i] = new_task_dir
                # if cp_files:
                #     os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
                #     shutil.copy(old_file_path, new_file_path)
    if cp_files:
        try:
            for task_dir in dataset.task_dirs:
                new_task_dir = re.sub(
                    r'^{0}'.format(re.escape(old_work_dir)),
                    new_work_dir, 
                    task_dir
                )
                copy_tree(task_dir, new_task_dir)
        except Exception as e:
            print(e)
    dataset.snapshot()



def viz_dataset(
    dataset,
    delta_height: float = 1,
    delta_width: float = 1
) -> t.Dict:
    """ Visualize the `Dataset` graph grouped by the task.

    Args:
        dataset (Dataset): A `Dataset` instance
        delta_height (float, optional): [description]. Defaults to 1.
        delta_width (float, optional): [description]. Defaults to 1.

    Returns:
        t.Dict: an ipycyto data
    """

    ipycyto_data = {}
    node_positions = {}
    ipycyto_data['nodes'] = []
    ipycyto_data['edges'] = []
    graph = nx.Graph(dataset.graph)
    layer_h = {}

    for node in graph.nodes:
        graph.nodes[node]['id'] = node
        layer = graph.nodes[node]['layer']
        if layer not in layer_h:
            layer_h[layer] = 0

        position = {
            'y': layer * delta_width,
            'x': layer_h[layer] * delta_height
        }
        layer_h[layer] += delta_height

        node_positions[node] = position

        ipycyto_data['nodes'].append(
            {'data': graph.nodes[node]}
        )

    for edge in graph.edges:
        start_id = edge[0]
        end_id = edge[1]
        edge_data = {
            'data': {
                'source': start_id,
                'target': end_id
            }
        }
        ipycyto_data['edges'].append(edge_data)

    cyto = ipycytoscape.CytoscapeWidget()
    cyto.graph.add_graph_from_json(ipycyto_data)
    cyto.set_style([
        {
            'css': {
                'background-color': '#ff2d00',
                'shape': 'circle',
                'width': '0.3',
                'height': '0.3',
                'border-color': 'rgb(0,0,0)',
                'border-opacity': 1.0,
                'border-width': 0.0,
                'color': '#4579e8',
                'background-fit': 'contain'
            },
            'selector': 'node'
        },
        {
            'css': {
                'width': 0.1,
                'target-arrow-shape': 'triangle'
            },
            'selector': 'edge'
        }
    ])
    cyto.set_layout(name='preset', positions=node_positions)
    return cyto


def collect_results(job) -> t.Tuple:
    """collect results from jobs

    Args:
        job (TaskBase): a task

    Returns:
        t.Tuple: a tuple of results for dataset.update_result
    """
    task = job['task']
    source = job['source']
    event_type = job['event_type']
    event_name = task.name
    outputs = task.run()
    done = task.done
    return (outputs, source, event_type, event_name, done)


def draw_graph(g: nx.Graph):
    plt.figure(1, figsize=(13,7))
    # nx.draw_networkx(g, ax=ax)
    # top = nx.bipartite.sets(g)[0]
    # pos = nx.bipartite_layout(g, top)
    pos = data_layout(g)

    nx.draw(
        g,
        pos,
        node_color='orange',
        node_size=600,
        with_labels = True,
        edge_cmap=plt.cm.Reds,
        font_size=8
    )
    plt.show()


def what_parser(trade_list):
    parsed_list = []
    for each_task_info in trade_list:
        trade_dict = dict()
        for key in each_task_info.keys():
            if key == 'task':
                class_name = type(each_task_info[key]).__name__.strip().split('.')[-1]  # AddLigandInfo
                hash_code = str(each_task_info[key]).split(' ')[-1].replace('>', '')  # 0x7f2abd9ba6d0
                trade_dict[key] = (class_name, hash_code)
            elif isinstance(each_task_info[key], (numbers.Number, str, list)):
                trade_dict[key] = each_task_info[key]
            else:
                print("key {0},type is {1},are skipped.".format(key, type(each_task_info[key])))
        parsed_list.append(trade_dict)
    return parsed_list


parser_factory = {
    'originator_id': str,
    'timestamp': str,
    'originator_version': int,
    'originator_topic': str,
    'what': what_parser
}


def parse_trade(x):
    event_type = type(x).__name__
    result_dict = {'event_type': event_type}
    for key in x.__dict__.keys():
        val = x.__dict__[key]
        result_dict[key] = parser_factory[key](val)
    return result_dict


def is_descendant(
    G: nx.DiGraph,
    node1: t.Any,
    node2: t.Any
) -> bool:
    """Check the whether two nodes are connected in a directed `Graph`
    return to `True` only when the second node is a descendant of the first node

    Args:
        G (nx.DiGraph): The directed Graph
        nodes (t.Iterable): `node_id` list

    Returns:
        bool: whether the second node is a descendant of the first node. 
    
    Examples:
    1 -> 2 -> 3
     ↘ 4 -> 5
    >>> descendants_connected(G, 1, 3)
    True
    
    >>> descendants_connected(G, 2, 1])
    False 

    >>> descendants_connected(G, 1, 5)
    True
    """
    assert isinstance(G, nx.DiGraph), "You should provide a directed `DiGraph`"
    return (
        node2 in nx.algorithms.descendants(G, node1)
    )


def is_descendants_connected_list(
    G: nx.DiGraph,
    nodes: t.Iterable
) -> bool:
    """Check the whether the nodes are connected in a directed `Graph`

    Args:
        G (nx.DiGraph): The directed Graph
        nodes (t.Iterable): `node_id` list

    Returns:
        bool: whether the nodes are connected by directed edges
    
    Examples:
    1 -> 2 -> 3
     ↘ 4 -> 5
    >>> descendants_connected(G, [1, 2, 3])
    True
    
    >>> descendants_connected(G, [2, 4, 1])
    False 

    >>> descendants_connected(G, [1, 4, 5])
    True
    """
    assert isinstance(G, nx.DiGraph), "You should provide a directed `DiGraph`"
    for i_node, node_id in enumerate(nodes):
        if i_node != 0:
            if not is_descendant(G, nodes[i_node - 1], node_id):
                return False
    return True


def get_common_children(
    G: nx.DiGraph,
    node1: t.Any,
    node2: t.Any
) -> t.List:
    """get children nodes of two nodes in a graph

    Args:
        G (nx.DiGraph): a `DiGraph`
        node1 (t.Any): first node
        node2 (t.Any): second node

    Returns:
        t.List: common children node ids.
    
    Examples:
    1 -> 2 -> 3
     ↘ 4 ->  5
    6 -> 7 ↗  

    >>> get_common_children(G, 4, 7)  # share 5
    [5]
     
    >>> get_common_children(G, 4, 6)  # 5 is a successor of 6 but not children
    []
    """
    assert isinstance(G, nx.DiGraph), "You should provide a directed `DiGraph`"
    children_of_node1 = list(G.neighbors(node1))
    children_of_node2 = list(G.neighbors(node2))
    return list(set(children_of_node1).intersection(set(children_of_node2)))


def share_same_children(
    G: nx.DiGraph,
    node1: t.Any,
    node2: t.Any
) -> bool:
    """Whether two nodes share the same children nodes in a graph

    Args:
        G (nx.DiGraph): a `DiGraph`
        node1 (t.Any): first node
        node2 (t.Any): second node

    Returns:
        bool: return to `True` only if two nodes share the same children node(s)
    
    Examples:
    1 -> 2 -> 3
     ↘ 4 ->  5
    6 -> 7 ↗  

    >>> share_same_children(G, 4, 7)  # share 5
    True
     
    >>> share_same_children(G, 4, 6)  # 5 is a successor of 6 but not children
    False
    """
    return any(get_common_children(G, node1, node2))


def get_each_pair_of_list(nodes: t.List) -> t.List[t.List]:
    """Get pairs of list of nodes

    Args:
        nodes (t.List): a list of node ids

    Returns:
        t.List[t.List]: the pairs of node ids
        
    Examples:
    >>> a = ['a', 'b', 'c', 'd']
    >>> get_each_pair_of_list(a)
    [['a', 'b'], ['b', 'c'], ['c', 'd']]
    """
    num_pairs = len(nodes) - 1
    indices = np.vstack([np.arange(num_pairs), np.arange(num_pairs) + 1]).T
    return np.array(nodes)[indices].tolist()


# @numba.jit
def joinmerge_two_dicts(
    dict_1: t.Dict,
    dict_2: t.Dict,
    merge_keys: t.Iterable,
    merge_func: t.Union[t.Callable, t.List[t.Callable]]
):
    """Merge two dict with some keys, values of `merge_keys` will be merged according to
    the `merge_func`, the returned `merged_dict` will only keep the `merge_keys`

    Args:
        dict_1 (t.Dict): the first `dict`
        dict_2 (t.Dict): the second `dicg`
        merge_keys (t.List): list of keys whose valued will be merged
        merge_func (t.Union[t.Callable, t.List[t.Callable]]): 
            a function to merge two elements or a list of `merge_funcs`

    Returns:
        t.Dict: the merged `dict`
    
    Examples:
    >>> dict_1 = {'a': 1, 'b': '1', 'c': [1, 2], 'd': 5}
    >>> dict_2 = {'b': '2', 'c': [3, 4], 'd': 6}
    
    >>> merge_two_dicts(
    ...     dict_1,
    ...     dict_2,
    ...     ['c', 'd'],
    ...     merge_func=lambda x, y: x + y, 
    ... )
    {'c': [1, 2, 3, 4], 'd': 11}
    
    >>> merge_two_dicts(
    ...     dict_1,
    ...     dict_2,
    ...     ['c', 'd'],
    ...     merge_func=[lambda x, y: x + y, lambda x, y: x * y]
    ... ) 
    {'c': [1, 2, 3, 4], 'd': 30}
    """
    assert (
        (
            isinstance(merge_func, t.Iterable)
            and len(merge_func) == len(merge_keys)
        ) or isinstance(merge_func, t.Callable)
    ), "You should provide a `function` or a list of functions with the same length with `merge_keys`"
    if isinstance(merge_func, t.Callable):
        merge_func = [merge_func] * len(merge_keys)

    merged_dict = {}
    for merge_key, func in zip(merge_keys, merge_func):
        merged_dict[merge_key] = func(
            dict_1[merge_key], dict_2[merge_key]
        )
    return merged_dict


def merge_two_dicts(
    dict_1: t.Dict,
    dict_2: t.Dict,
    merge_keys: t.Iterable,
    merge_func: t.Union[t.Callable, t.List[t.Callable]]
) -> t.Dict:
    """Merge two dict with the same keys, the `ref_keys` will keep values im the
    first `dict`, other values of the `merge_keys` will be merged according to
    the `merge_func`

    Args:
        dict_1 (t.Dict): the first `dict`
        dict_2 (t.Dict): the second `dicg`
        merge_keys (t.List): list of keys whose valued will be merged, values of
            other keys will be the same as the values in the first dict.
        merge_func (t.Union[t.Callable, t.List[t.Callable]]): 
            a function to merge two elements or a list of `merge_funcs`

    Returns:
        t.Dict: the merged `dict`
    
    Examples:
    >>> dict_1 = {'a': 1, 'b': '1', 'c': [1, 2], 'd': 5}
    >>> dict_2 = {'b': '2', 'c': [3, 4], 'd': 6}
    
    >>> merge_two_dicts(
    ...     dict_1,
    ...     dict_2,
    ...     ['c', 'd'],
    ...     merge_func=lambda x, y: x + y, 
    ... )
    {'a': 1, 'b': '2', 'c': [1, 2, 3, 4], 'd': 11}
    
    >>> merge_two_dicts(
    ...     dict_1,
    ...     dict_2,
    ...     ['c', 'd'],
    ...     merge_func=[lambda x, y: x + y, lambda x, y: x * y]
    ... ) 
    {'a': 1, 'b': '2', 'c': [1, 2, 3, 4], 'd': 30}
    """

    ref_keys = list(set(dict_1.keys()) - set(merge_keys))
    # values of the ref keys in the first dict will be the default values
    merged_dict = {}
    for ref_key in ref_keys:
        merged_dict[ref_key] = dict_1[ref_key]

    join_dict = joinmerge_two_dicts(
        dict_1, dict_2,
        merge_keys=merge_keys,
        merge_func=merge_func
    )
    merged_dict.update(join_dict)
    return merged_dict


def merge_dicts_in_list(
    dict_list: t.List[t.Dict],
    merge_keys: t.Iterable,
    merge_func: t.Union[t.Callable, t.List[t.Callable]]
) -> t.Dict:
    """Merge `dicts` in a `list` with given `ref_keys`, `merge_keys` and `merge_func`

    Args:
        dict_list (t.List[t.Dict]): a `list` of `dicts` with same keys
        ref_keys (t.Iterable): a list of ref keys whose value will be the values
            in the first dict of the list
        merge_keys (t.Iterable): list of keys whose valued will be merged 
        merge_func (t.Union[t.Callable, t.List[t.Callable]]): 
            a function to merge two elements or a list of `merge_funcs`

    Returns:
        t.Dict: the merged `dict`
    
    Examples:
    >>> dict_1 = {'a': 1, 'b': '1', 'c': [1, 2], 'd': 5}
    >>> dict_2 = {'b': '2', 'c': [3, 4], 'd': 6}
    >>> ls = [dict_1, dict_1, dict_2]
    >>> merge_dicts_in_list(
    ...     ls,
    ...     ['c', 'd'],
    ...     merge_func=[lambda x, y: x + y, lambda x, y: x * y],
    ... ) 
    {'a': 1, 'b': '1', 'c': [1, 2, 1, 2, 3, 4], 'd': 150}
    """
    merged_dict = {}
    for dict_id, each_dict in enumerate(dict_list):
        if dict_id == 0:
            merged_dict = each_dict
        else:
            merged_dict = merge_two_dicts(
                merged_dict,
                each_dict,
                merge_keys=merge_keys,
                merge_func=merge_func
            )
    return merged_dict


# @numba.jit
def merge_dicts_withref_in_list(
    dict_list: t.List[t.Dict],
    ref_keys: t.Iterable,
    merge_keys: t.Iterable,
    merge_funcs: t.Union[t.Callable, t.List[t.Callable]]
) -> t.Dict:
    """Merge dicts in a list group by the `ref_keys`

    Args:
        dict_list (t.List[t.Dict]): the list of dicts
        ref_keys (t.Iterable): ref keys
        merge_keys (t.Iterable): keys to merge
        merge_func (t.Union[t.Callable, t.List[t.Callable]]): 
            a function to merge two elements or a list of `merge_funcs`

    Returns:
        t.Dict: The returned dict with `HashableDict` as keys.
    
    Examples:
    >>> dict_1 = {'a': 1, 'b': '1', 'c': [1, 2], 'd': 5}
    >>> dict_2 = {'a': 1, 'b': '2', 'c': [3, 4], 'd': 6}
    >>> ls = [dict_1, dict_1, dict_2, dict_2]
    >>> merge_dicts_withref_in_list(
    ...     ls,
    ...     ['a', 'b'],
    ...     ['c', 'd'],
    ...     [lambda x, y: x + y, lambda x, y: x * y]
    ... )
    {{'a': 1, 'b': '1'}: {'c': [1, 2, 1, 2], 'd': 25},
    {'a': 1, 'b': '2'}: {'c': [3, 4, 3, 4], 'd': 36}}
    """
    if isinstance(merge_funcs, t.Callable):
        merge_funcs = [merge_funcs] * len(merge_keys)

    ref_dict = {}
    # join_dict = {}
    for source_dict in dict_list:
        keyof_ref_dict = HashableDict({})
        for ref_key in ref_keys:
            keyof_ref_dict[ref_key] = source_dict[ref_key]

        if keyof_ref_dict in ref_dict:
            corrected_merge_funcs = []
            for merge_func in merge_funcs:
                # if dict_idx == 0: 
                #     corrected_merge_funcs.append(merge_as_list_with_init)
                # elif dict_idx != 0:
                if merge_func == merge_as_list:
                    corrected_merge_funcs.append(merge_as_list_without_init)
                else:
                    corrected_merge_funcs.append(merge_func)
            # print(corrected_merge_funcs)
            merged_dict = joinmerge_two_dicts(
                ref_dict[keyof_ref_dict], source_dict,
                merge_keys=merge_keys,
                merge_func=corrected_merge_funcs
            )
            ref_dict[keyof_ref_dict].update(merged_dict)
        else:
            ref_dict[keyof_ref_dict] = {}
            for merge_key, merge_func in zip(merge_keys, merge_funcs):
                if merge_func == merge_as_list:
                    ref_dict[keyof_ref_dict][merge_key] = [source_dict[merge_key]]
                else:
                    ref_dict[keyof_ref_dict][merge_key] = source_dict[merge_key]
    return ref_dict


def merge_dicts_aslist_withref_in_list(
    dict_list: t.List[t.Dict],
    ref_keys: t.Iterable,
    merge_keys: t.Iterable,
) -> t.Dict:
    """Merge dicts in a list group by the `ref_keys`

    Args:
        dict_list (t.List[t.Dict]): the list of dicts
        ref_keys (t.Iterable): ref keys
        merge_keys (t.Iterable): keys to merge

    Returns:
        t.Dict: The returned dict with `HashableDict` as keys.
    
    Examples:
    >>> dict_1 = {'a': 1, 'b': '1', 'c': [1, 2], 'd': 5}
    >>> dict_2 = {'a': 1, 'b': '2', 'c': [3, 4], 'd': 6}
    >>> ls = [dict_1, dict_1, dict_2, dict_2]
    >>> merge_dicts_aslist_withref_in_list(
    ...     ls,
    ...     ['a', 'b'],
    ...     ['c', 'd'],
    ... )
    {{'a': 1, 'b': '1'}: {'c': [[1, 2], [1, 2]], 'd': [5, 5]},
    {'a': 1, 'b': '2'}: {'c': [[3, 4], [3, 4]], 'd': [6, 6]}}
    """

    ref_dict = {}
    merge_funcs = [merge_as_list_without_init] * len(merge_keys)
    # join_dict = {}
    for source_dict in dict_list:
        keyof_ref_dict = HashableDict({})
        for ref_key in ref_keys:
            keyof_ref_dict[ref_key] = source_dict[ref_key]

        if keyof_ref_dict in ref_dict:
            merged_dict = joinmerge_two_dicts(
                ref_dict[keyof_ref_dict], source_dict,
                merge_keys=merge_keys,
                merge_func=merge_funcs
            )
            ref_dict[keyof_ref_dict].update(merged_dict)
        else:
            ref_dict[keyof_ref_dict] = {}
            for merge_key in merge_keys:
                ref_dict[keyof_ref_dict][merge_key] = [source_dict[merge_key]]
    return ref_dict


class HashableDict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def ifp_dict_to_dataframe(ifp_dict_series):
    """
    pandas.core.series.Series, a series of ifp dictionaries
    for example:
    :param ifp_dict_series:
     0    {'HY:ILE1...Core': 0, 'HY:ILE1...R1': 0}
     1    {'HY:ILE1...Core': 0, 'HY:ILE1...R1': 0}
     Name: ifp_dict_series, dtype: object
    """

    df = pd.DataFrame.from_dict(ifp_dict_series.to_dict(), orient='index')
    return df


# @numba.jit
def merge_as_list(
    item_1: t.Any,
    item_2: t.Any,
    init_list: bool
) -> t.List:
    if init_list:
        return [item_1, item_2]
    else:
        item_1.append(item_2)
        return item_1


def merge_as_list_with_init(
    item1,
    item2
):
    return merge_as_list(item1, item2, init_list=True)


def merge_as_list_without_init(item1, item2):
    return merge_as_list(item1, item2, init_list=False)


# merge_as_list_with_init = partial(merge_as_list, init_list=True)
# merge_as_list_without_init = partial(merge_as_list, init_list=False)


def append_connection_modes(
    input_nodes: t.List[t.List],
    connection_modes: t.List[str]
):
    input_node_groups = []
    connection_mode_list = []
    
    assert len(connection_modes) == len(input_nodes) - 1

    num_node_groups = len(input_nodes)

    for i in range(num_node_groups):
        if i == 0:
            input_node_groups.append(input_nodes[i])
        else:
            if connection_modes[i - 1] != 'append':
                input_node_groups.append(input_nodes[i])
                connection_mode_list.append(connection_modes[i - 1])
            else:
                input_node_groups[-1].extend(input_nodes[i])
    return input_nodes, connection_mode_list


def append_connection_list(
    input_nodes: t.List[t.List],
    connection_list: t.List[t.List]  # [[0, 2], [3, 4, 5]]
):
    """Append node in a group of input nodes

    Args:
        input_nodes (t.List[t.List]): list of input node combinations
        connection_dict (t.Dict): connection dict with `append` in keys

    Returns:
        t.Tuple[t.List, t.Dict]: the new input nodes and layer mapping
    
    Examples:
    >>> input_nodes = [
    ...     [0, 1, 2],
    ...     [3, 4, 5],
    ...     [6, 7, 8],
    ...     [9, 10, 11],
    ...     [12, 13, 14],
    ...     [15, 16, 17],
    ...     [18, 19, 20]
    ... ]
    >>> connection_dict = [[2, 0], [3, 4, 6]]
    >>> append_connection_dicts(input_nodes, connection_dict)
    (
        [
            [6, 7, 8, 0, 1, 2],
            [9, 10, 11, 12, 13, 14, 18, 19, 20],
            [3, 4, 5],
            [15, 16, 17]
        ],
        {2: 0, 0: 0, 3: 1, 4: 1, 6: 1, 1: 2, 5: 3}
    )
    """
    # assert 'append' in connection_list, "The append key is not in connection_dict"
    num_appends = len(connection_list)

    id_map_dict = {}
    new_input_nodes = []

    # append ids
    for append_idgroup_idx, append_idgroup in enumerate(connection_list):
        this_group_ids = []
        for append_id in append_idgroup:
            id_map_dict[append_id] = append_idgroup_idx
            this_group_ids.extend(input_nodes[append_id])
        new_input_nodes.append(this_group_ids)
    
    # other ids
    append_idgroups_flat = itertools.chain.from_iterable(connection_list)
    not_append_ids = list(set(range(len(input_nodes))) - set(append_idgroups_flat))
    for i, not_append_id in enumerate(not_append_ids):
        new_input_nodes.append(input_nodes[not_append_id])
        id_map_dict[not_append_id] = num_appends + i

    return new_input_nodes, id_map_dict


def all_simple_paths_for_pair_list(
    graph: nx.DiGraph,
    source_nodes: t.List,
    target_nodes: t.List
):
    """get path dictionary for a pair of node lists

    Args:
        graph: nx.DiGraph()
        source_nodes: list of node ids
        target_nodes: list of node ids

    Returns:
        dict[source][target] = [source, node1, node2, ..., target]
    
    Examples:
    >>> test_g = nx.DiGraph()
    1  →  2  →  3  →  6  →  7  
     ↘        ↗   ↘   ↑   ↗ 
      4  →  5         8
       
    >>> all_simple_paths_for_pair_list(test_g, [1, 4], [3, 6])
    {
        1: {
            3: [[1, 2, 3], [1, 4, 5, 3]],
            6: [[1, 2, 3, 6], [1, 2, 3, 8, 6], [1, 4, 5, 3, 6], [1, 4, 5, 3, 8, 6]]
        },
        4: {
            3: [[4, 5, 3]],
            6: [[4, 5, 3, 6], [4, 5, 3, 8, 6]]
        }
    }

    """
    paths = {}
    for source in source_nodes:
        paths[source] = {}
        for target in target_nodes:
            paths[source][target] = [
                p for p in nx.simple_paths.all_simple_paths(graph, source, target)
            ]

    return paths


def connected_components(
    graph: nx.DiGraph,
    source_nodes: t.List,
    target_nodes: t.List
):
    """get connnected pairs as a path for a pair of node lists

    Args:
        graph: nx.DiGraph()
        source_nodes: list of node ids
        target_nodes: list of node ids

    Returns:
        dict[source][target] = [source, target]
    
    Examples:
    >>> test_g = nx.DiGraph()
    1  →  2  →  3  →  6  →  7  
     ↘        ↗   ↘   ↑   ↗ 
      4  →  5         8
    >>> connected_components(g, [4, 3, 2], [1, 7, 8]) 
    {
        4: {
            7: [(4, 7)],
            8: [(4, 8)]
        },
        3: {
            7: [(3, 7)],
            8: [(3, 8)]
        },
        2: {
            7: [(2, 7)],
            8: [(2, 8)]
        }
    }

    """
    paths = {}
    for source in source_nodes:
        paths[source] = {}
        for target in target_nodes:
            if nx.has_path(graph, source, target):
                paths[source][target] = [(source, target)]

    return paths


def connect_path_dict(
    d1: t.Dict,
    d2: t.Dict
):
    """get path dictionary for two path dict, merge two paths 

    Args:
        d1 (t.Dict): 
        d2 (t.Dict):
    Returns:
        dict[source][target] = [source, node1, node2, ..., target]
    
    Examples:
    >>> test_g = nx.DiGraph()
    1  →  2  →  3  →  6  →  7  
     ↘        ↗   ↘   ↑   ↗ 
      4  →  5 → 9  →  8 
    >>> evidences = all_simple_paths_for_pair_list(test_g, [1], [3, 9])
    # evideces = {1: {3: [[1, 2, 3], [1, 4, 5, 3]], 9: [[1, 4, 5, 9]]}}
    >>> connect_path_dict(
        evidences, 
        all_simple_paths_for_pair_list(test_g, [3, 9], [6, 7])
    )

    {1: defaultdict(list,
             {6: [[1, 2, 3, 6],
               [1, 2, 3, 8, 6],
               [1, 4, 5, 3, 6],
               [1, 4, 5, 3, 8, 6],
               [1, 4, 5, 9, 8, 6]],
              7: [[1, 2, 3, 6, 7],
               [1, 2, 3, 8, 6, 7],
               [1, 2, 3, 8, 7],
               [1, 4, 5, 3, 6, 7],
               [1, 4, 5, 3, 8, 6, 7],
               [1, 4, 5, 3, 8, 7],
               [1, 4, 5, 9, 8, 6, 7],
               [1, 4, 5, 9, 8, 7]]})}

    """
    paths = {}
    for s1 in d1:
        paths[s1] = defaultdict(list)
        for t1 in d1[s1]:
            paths1 = d1[s1][t1]
            for t2 in d2[t1]:
                paths2 = d2[t1][t2]
                paths[s1][t2] += [
                    list(itertools.chain(item[0][:-1], item[1]))
                    for item in itertools.product(paths1, paths2)
                ]

    return paths


def defaultdict_of_list():
    return defaultdict(list)


def induced_undirected_graph(
    g: nx.DiGraph,
    nodes: t.Iterable
) -> nx.Graph:
    """Adopted from xumin, get induced proxy undirected graph of a `Graph`

    Args:
        g (nx.Graph): the source graph
        nodes (t.Iterable): a list of node ids

    Returns:
        nx.Graph: the proxy induced graph
    """
    new_g = nx.DiGraph(g)
    nodes = set(nodes)
    
    # delete all in-edges from the "MODEL" and "HYPOTHESIS" layers
    # to_delete_edges = []
    # for e in new_g.edges:
    #     _nt_i = e[0].split('.')[1]
    #     _nt_j = e[1].split('.')[1]
    #     if (
    #         (_nt_j == 'HYPOTHESIS' and _nt_i != 'HYPOTHESIS') 
    #         or (_nt_j == 'MODEL' and _nt_i == 'LIGAND')
    #     ):
    #         to_delete_edges.append(e)

    # new_g.remove_edges_from(to_delete_edges)

    for n in sorted(
        set(g.nodes) - nodes,
        key=lambda x: g.in_degree(x),
        reverse=True
    ):
        #  TODO: 不知道会不会有问题，有问题再解决

        out_edges = new_g.out_edges(n)
        in_edges = new_g.in_edges(n)
        out_nodes = []
        
        for _, i in out_edges:
            for j, _ in in_edges:
                if i != j:
                    new_g.add_edge(j, i)
            out_nodes.append(i)
        new_g.remove_node(n)
        
        for i in range(len(out_nodes) - 1):
            for j in range(i + 1, len(out_nodes)):
                if out_nodes[i] != out_nodes[j]:
                    new_g.add_edge(out_nodes[i], out_nodes[j])
                    new_g.add_edge(out_nodes[j], out_nodes[i])

    new_g = new_g.to_undirected()

    return new_g


def level_changed(level_list: t.List) -> bool:
    num_levels = len(level_list)
    level_diffs = [
        level_list[i + 1] - level_list[i] for i in range(num_levels - 1)
    ]
    # print(level_diffs)
    if all(x >= 0 for x in level_diffs):
        return False
    elif all(x <= 0 for x in level_diffs):
        if level_list[-1] == 2:
            return False
        elif (
            sum([x == 1 for x in level_list]) <= 1
            and sum([x == 0 for x in level_list]) <= 1
        ):
            return False
        else:
            return True
    else:
        return True


def up_and_flat(level_diffs):
    num_diffs = len(level_diffs)
    if num_diffs == 1:
        return False
    diff_tag = -1
    for i, level_diff in enumerate(level_diffs):
        if i == 0:
            if level_diff >= 0:
                return False
        if diff_tag == -1:
            if level_diff == 0:
                diff_tag = 0
            elif level_diff > 0:
                return False
        if diff_tag == 0:
            if level_diff != 0:
                return False
    return True 


def data_layout(g):
    layout_dict = {}
    layout_dict_dict = {}
    # num_ancestors = []
    max_n_ac = 0
    for node in g.nodes():
        layout_dict_dict[node] = {}
        if node.split('.')[1] == 'HYPOTHESIS':
            layout_dict_dict[node]['y'] = np.random.uniform(0.7, 0,9)
        elif node.split('.')[1] == 'MODEL':
            layout_dict_dict[node]['y'] = np.random.uniform(-0.1, 0.7)
        else:
            layout_dict_dict[node]['y'] = np.random.uniform(-0.9, -0.1)
        layout_dict_dict[node]['n_ac'] = len(ancestors(g, node))
        if layout_dict_dict[node]['n_ac'] > max_n_ac:
            max_n_ac = layout_dict_dict[node]['n_ac']
    for node in layout_dict_dict:
        layout_dict_dict[node]['x'] = (
            layout_dict_dict[node]['n_ac']
            / (max_n_ac + 1)
            + np.random.uniform(-0.05, 0.05)
        )
        layout_dict[node] = np.array(
            [layout_dict_dict[node]['x'], layout_dict_dict[node]['y']]
        )
    return layout_dict
