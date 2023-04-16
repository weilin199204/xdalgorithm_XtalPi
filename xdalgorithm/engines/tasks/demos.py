from random import (
    random,
    randint,
)
from copy import deepcopy

import numpy as np
import pandas as pd
import networkx as nx

from .base import UNDEFINED_PARAMETER
from .base import TaskBase
from .base import CollectiveTaskBase

import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict

class DemoProteinFixer(TaskBase):
    def __init__(self, name: str = 'demo_protein'):
        super().__init__(name)
        self.config_template = {
            'pdb_file_name': UNDEFINED_PARAMETER
        }
    
    def run(self):
        
        return [
            {'protein_conf_name': f'protein_conf_{str(i)}'}
            for i in range(10)
        ]


class DemoGrid(TaskBase):
    def __init__(self, name: str = 'grid'):
        super().__init__(name)
        self.config_template = {
            'pdb_file_name': UNDEFINED_PARAMETER,
        }

    def run(self):
        return [
            {'grid_file': self.config_template['pdb_file_name']}
        ]


class DemoLigand(TaskBase):
    def __init__(self, name: str = 'ligand'):
        super().__init__(name)
        self.config_template = {
            'ligand_file': UNDEFINED_PARAMETER
        }
    
    def run(self):
        return [
            {
                'mol_name': f"mol_{i:05d}",
                'aff': random()
            } for i in range(10)
        ]


class DemoConf(TaskBase):
    def __init__(self, name: str = 'conf'):
        super().__init__(name)
        self.config_template = {
            'mol_name': UNDEFINED_PARAMETER
        }
    
    def run(self):
        num_confs = randint(1, 2)
        mol_name = self.config_template['mol_name']
        result_list = []
        for i in range(num_confs):
            conf_name = mol_name + '_' + str(i)
            result_dict = {'conf_name': conf_name}
            result_list.append(result_dict)
        return result_list


class DemoCore(CollectiveTaskBase):
    def __init__(self, name: str = 'core'):
        super().__init__(name)
        self.config_template = {
            'mol_name': UNDEFINED_PARAMETER,
            'rgroup_df': UNDEFINED_PARAMETER
        }
    
    def collect_config(self, task: "TaskBase"):
        if isinstance(self.config_template['mol_name'], str):
            self.config_template['mol_name'] = [self.config_template['mol_name']]
        self.config_template['mol_name'].append(task.config_template['mol_name'])
    
    def run(self):
        result_list = []
        for i, mol_name in enumerate(self.config_template['mol_name']):
            result_dict = {}
            result_dict['output_ids'] = [i]
            result_dict['core'] = self.config_template['rgroup_df'][self.config_template['rgroup_df']['mol_name']==mol_name].iloc[0, 1]
            result_list.append(result_dict)
            
        return result_list


class DemoIFPLabel(CollectiveTaskBase):
    def __init__(self, name: str = 'ifp_labels'):
        super().__init__(name)
        self.config_template = {
            "protein_conf_name": UNDEFINED_PARAMETER,
            "mol_name": UNDEFINED_PARAMETER,
            "core": UNDEFINED_PARAMETER 
        }
    
    def collect_config(self, task: "TaskBase"):
        if isinstance(self.config_template, dict):
            self.config_template = [self.config_template]
        self.config_template.append(task.config_template)

    def run(self):
        return [
            
            {
                'output_ids': list(range(len(self.config_template))),
                'ifp_labels': 'label'
            }
        ]


class DemoDock(CollectiveTaskBase):
    def __init__(self, name: str = 'dock'):
        super().__init__(name)
        self.config_template = {
            'protein_conf_name': UNDEFINED_PARAMETER,
            'grid_file': UNDEFINED_PARAMETER,
            'conf_name': UNDEFINED_PARAMETER
        }
    
    def collect_config(self, task: "TaskBase"):
        if isinstance(self.config_template, dict):
            self.config_template = [self.config_template]
        self.config_template.append(task.config_template)
        
    def run(self):
        result_list = []
        for i, config in enumerate(self.config_template):
            for j in range(3):
                result_dict = {}
                result_dict['output_ids'] = [i]
                result_dict['docked_pose'] = f"docked_{config['grid_file']}_{config['conf_name']}_{j}"
                result_list.append(result_dict)
                
        return result_list

class DemoGenericCore(CollectiveTaskBase):
    def __init__(self, name: str = 'gc'):
        super().__init__(name)
        self.config_template = {
            'core': UNDEFINED_PARAMETER
        }

    def collect_config(self, task: "TaskBase"):
        if isinstance(self.config_template, dict):
            self.config_template = [self.config_template]
        self.config_template.append(task.config_template)
    
    def run(self):
        max_patt=None
        for config in self.config_template:
            if max_patt is None:
                max_patt = set(config['core'])
            else:
                max_patt &= set(config['core'])
                
        return [{'generic_core': max_patt}]


class DemoIFP(CollectiveTaskBase):
    def __init__(self, name: str = 'ifp'):
        super().__init__(name)
        self.config_template = {
            'protein_conf_name': UNDEFINED_PARAMETER,
            'conf_name': UNDEFINED_PARAMETER,
            'generic_core': UNDEFINED_PARAMETER
        }
    
    def collect_config(self, task: "TaskBase"):
        if isinstance(self.config_template, dict):
            self.config_template = [self.config_template]
        self.config_template.append(task.config_template)
    
    def run(self):
        result_list = []
        for i, config in enumerate(self.config_template):
            result_dict = deepcopy(config)
            result_dict['output_ids'] = [i]
            result_dict['ifp'] = f"ifp_{str(i)}"
            result_list.append(result_dict)
        return result_list


class DemoIntra(CollectiveTaskBase):
    def __init__(self, name: str = 'intra'):
        super().__init__(name)
        self.config_template = {
            'mol_name': UNDEFINED_PARAMETER,
            'protein_conf_name': UNDEFINED_PARAMETER,
            'docked_pose': UNDEFINED_PARAMETER
        }
        self.i = 0
    
    def collect_config(self, task: "TaskBase"):
        if isinstance(self.config_template, dict):
            self.config_template = [
                {
                    'mol_name': self.config_template['mol_name'],
                    'protein_conf_name': self.config_template['protein_conf_name'],
                    'docked_pose': [self.config_template['docked_pose']],
                    'ids': [self.i]
                }
            ]
            self.i += 1
            
        for config in self.config_template:
            found=False
            if task.config_template['mol_name']==config['mol_name'] and \
               task.config_template['protein_conf_name']==config['protein_conf_name']:
                config['docked_pose'].append(task.config_template['docked_pose'])
                config['ids'].append(self.i)
                found=True
                
        if not found:
            self.config_template.append({
                'mol_name': task.config_template['mol_name'],
                'protein_conf_name': task.config_template['protein_conf_name'],
                'docked_pose': [task.config_template['docked_pose']],
                'ids': [self.i]
            })
        
        self.i += 1
    
    def run(self):
        print(self.config_template)
        result_list = []
        
        for config in self.config_template:
            n_clusters = np.random.randint(1,3)
            _r = [defaultdict(list) for _ in range(n_clusters)]
            
            for i,j in enumerate(np.random.randint(n_clusters, size=len(config['docked_pose']))):
                _r[j]['output_ids'].append(config['ids'][i])
                if not 'docked_pose' in _r[j].keys():
                    _r[j]['docked_pose'] = config['docked_pose'][i]
                print(_r)
                    
            result_list += [x for x in _r if x]
                
        return result_list


class DemoInter(CollectiveTaskBase):
    def __init__(self, name: str = 'inter'):
        super().__init__(name)
        self.config_template = {
            'generic_core': UNDEFINED_PARAMETER,
            'protein_conf_name': UNDEFINED_PARAMETER,
            'docked_pose': UNDEFINED_PARAMETER
        }
        self.i = 0
    
    def collect_config(self, task: "TaskBase"):
        if isinstance(self.config_template, dict):
            self.config_template = [
                {
                    'protein_conf_name': self.config_template['protein_conf_name'],
                    'docked_pose': [self.config_template['docked_pose']],
                    'ids': [self.i]
                }
            ]
            self.i += 1
            
        for config in self.config_template:
            found=False
            if task.config_template['protein_conf_name']==config['protein_conf_name']:
                config['docked_pose'].append(task.config_template['docked_pose'])
                config['ids'].append(self.i)
                found=True
                
        if not found:
            self.config_template.append({
                'protein_conf_name': task.config_template['protein_conf_name'],
                'docked_pose': [task.config_template['docked_pose']],
                'ids': [self.i]
            })
        
        self.i += 1
    
    def run(self):
        print(self.config_template)
        n_clusters = 5
        result_list = [defaultdict(list) for _ in range(n_clusters)]
        
        for config in self.config_template:
            n = len(config['docked_pose'])
            for i,j in enumerate(np.random.randint(n_clusters, size=n)):
                result_list[j]['output_ids'].append(config['ids'][i])
                if not 'cluster_id' in result_list[j].keys():
                    result_list[j]['cluster_id'] = j
        
        return result_list


class DemoSCNet(CollectiveTaskBase):
    _type = 'MODEL'

    def __init__(self, name: str = 'scnet'):
        super().__init__(name)
        self.config_template = {
            'mol_name': UNDEFINED_PARAMETER
        }
        
    def collect_config(self, task: "TaskBase"):
        if isinstance(self.config_template['mol_name'], str):
            self.config_template['mol_name'] = [self.config_template['mol_name']]
        self.config_template['mol_name'].append(task.config_template['mol_name'])
    
    def run(self):
        rgroup_df = pd.DataFrame(data={'mol_name':self.config_template['mol_name'],
                                     'core': ['core_%d' %i for i in range(len(self.config_template['mol_name']))]})
        graph = nx.path_graph(len(self.config_template))
        
        return[{
            'snet': graph,
            'rgroup_df': rgroup_df
        }]


class DemoHyp(CollectiveTaskBase):
    _type = 'HYPOTHESIS'

    def __init__(self, name: str = 'hyp'):
        super().__init__(name)
        self.config_template = {
            'cluster_rank': UNDEFINED_PARAMETER
        }
    
    def run(self):
        output_ids = list(range(len(self.config_template)))
        cluster_rank = 0
        return[{
            'output_ids': output_ids,
            'cluster_rank': cluster_rank
        }]


class DemoDominantCluster(CollectiveTaskBase):
    def __init__(self, name: str = 'dominant_cluster'):
        super().__init__(name)
        self.config_template = {
            'cluster_id': UNDEFINED_PARAMETER,
            'protein_conf_name': UNDEFINED_PARAMETER,
            'mol_name': UNDEFINED_PARAMETER,
            'docked_pose': UNDEFINED_PARAMETER,
            'dominant_cutoff': UNDEFINED_PARAMETER
        }
        self.i = 0
        self.cutoff = None
    
    def collect_config(self, task: "TaskBase"):
        if isinstance(self.config_template, dict):
            self.cutoff = self.config_template['dominant_cutoff']
            self.config_template = [
                {
                    'mol_name': self.config_template['mol_name'],
                    'protein_conf_name': self.config_template['protein_conf_name'],
                    'cluster_id': self.config_template['cluster_id'],
                    'docked_pose': [self.config_template['docked_pose']],
                    'ids': [self.i]
                }
            ]
            self.i += 1
            
        
        found=False
        for config in self.config_template:
            if task.config_template['mol_name']==config['mol_name'] and \
                task.config_template['protein_conf_name']==config['protein_conf_name'] and \
                task.config_template['cluster_id']==config['cluster_id']:
                config['docked_pose'].append(task.config_template['docked_pose'])
                config['ids'].append(self.i)
                found=True
                
        if not found:
            self.config_template.append({
                'mol_name': task.config_template['mol_name'],
                'protein_conf_name': task.config_template['protein_conf_name'],
                'cluster_id': task.config_template['cluster_id'],
                'docked_pose': [task.config_template['docked_pose']],
                'ids': [self.i]
            })
        
        self.i += 1
    
    def run(self):
        print(self.config_template, self.cutoff)
        all_mols = np.unique([x['mol_name'] for x in self.config_template])
        all_proteins = np.unique([x['protein_conf_name'] for x in self.config_template])
        all_clusters = np.unique([x['cluster_id'] for x in self.config_template])
#         print(all_mols, all_proteins, all_clusters)
        
        mol_matrix = {m: np.zeros((len(all_proteins), len(all_clusters))) for m in all_mols}
        for config in self.config_template:
            m = config['mol_name']
            p = config['protein_conf_name']
            c = config['cluster_id']
            
            mol_matrix[m][np.where(all_proteins==p)[0][0]][np.where(all_clusters==c)[0][0]] += len(config['docked_pose'])
        
        dominant_clusters = [[] for _ in range(len(all_clusters))]
        possess_clusters = [[] for _ in range(len(all_clusters))]
        
        for m in mol_matrix:
            sum_of_rows = mol_matrix[m].sum(axis=1)
            mol_matrix[m] /= sum_of_rows[:, np.newaxis]
            
            protein=None; cluster=None; freq = 0.
            
            for i, p in enumerate(all_proteins):
                ci, max_freq = np.argmax(mol_matrix[m][i,:]), np.max(mol_matrix[m][i,:])
                if max_freq > freq:
                    protein = p
                    cluster = ci
                    freq = max_freq
                    
            if max_freq > self.cutoff:
                dominant_clusters[cluster].append((m, protein))
            else:
                possess_clusters[cluster].append((m, protein))
            
#         print(dominant_clusters)
            
        result_list = []
        for i, item in enumerate(dominant_clusters):
            if len(item)==0: continue
                
            r = defaultdict(list)
            for m, p in item:
                for config in self.config_template:
                    if config['mol_name'] == m and config['protein_conf_name'] == p \
                    and config['cluster_id'] == i:
                        r['output_ids'] += config['ids']
                        r['dominated_mols'].append(m)
                        
            r['cluster_id'] = i
            result_list.append(r)
            
        for r in result_list:
            ci = r['cluster_id']
            for m, p in possess_clusters[ci]:
                for config in self.config_template:
                    if config['mol_name'] == m and config['protein_conf_name'] == p \
                    and config['cluster_id'] == ci:
                        r['possessed_pose'].append(config['docked_pose'][0])
                        
        return result_list


class DemoProposeForMD(CollectiveTaskBase):
    def __init__(self, name: str = 'propose'):
        super().__init__(name)
        self.config_template = {
            'cluster_id': UNDEFINED_PARAMETER,
            'dominated_mols': UNDEFINED_PARAMETER,
            'possessed_pose': UNDEFINED_PARAMETER,
            'docked_pose': UNDEFINED_PARAMETER,
            'budget': UNDEFINED_PARAMETER
        }
        self.budget = 0
        self.i = 0
    
    def collect_config(self, task: "TaskBase"):
        if isinstance(self.config_template, dict):
            self.budget = self.config_template['budget']
            self.config_template = [
                {
                    'cluster_id': self.config_template['cluster_id'],
                    'dominated_mols': self.config_template['dominated_mols'],
                    'possessed_pose': self.config_template['possessed_pose'],
                    'docked_pose': [self.config_template['docked_pose']],
                    'ids': [self.i]
                }
            ]
            self.i += 1
            
        
        found=False
        for config in self.config_template:
            if task.config_template['cluster_id']==config['cluster_id']:
                config['docked_pose'].append(task.config_template['docked_pose'])
                config['ids'].append(self.i)
                found=True
                
        if not found:
            self.config_template.append({
                'cluster_id': task.config_template['cluster_id'],
                'dominated_mols': task.config_template['dominated_mols'],
                'possessed_pose': task.config_template['possessed_pose'],
                'docked_pose': [task.config_template['docked_pose']],
                'ids': [self.i]
            })
        
        self.i += 1
        
    def run(self):
        print(self.config_template)        
        
        n_votes = [len(x['dominated_mols']) for x in self.config_template]
        prob = np.array(n_votes)/sum(n_votes)
        sample_ids = np.random.choice(len(self.config_template), self.budget, p=prob)
        
        result_list = []
        selected = set()
        
        for i in sample_ids:
            pose_name = np.random.choice(self.config_template[i]['possessed_pose'], 1)[0]
            if not pose_name in selected:
                r = {}
                selected.add(pose_name)
                j = self.config_template[i]['docked_pose'].index(pose_name)
                r['output_ids'] = [self.config_template[i]['ids'][j]]
                r['proposed_pose'] = [pose_name]
                
                result_list.append(r)
        
        return result_list