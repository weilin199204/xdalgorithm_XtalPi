"""
`TaskBase`
Your can write your own tasks to use in the `Dataset` which should return
a `list` of `dict` in the `run()` method.
"""
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from collections import Counter

__all__ = [
    "UNDEFINED_PARAMETER",
    "TaskBase",
    "CollectiveTaskBase",
]

UNDEFINED_PARAMETER = 'UNDEFINED'


class TaskBase(ABC):
    def __init__(self, name: str):
        self.name = name
        self.config_template = {}
        self.done: bool = False

    def update_config(self, **kwargs):
        for key in self.config_template:
            if key in kwargs:
                self.config_template[key] = kwargs[key]
     
    @abstractmethod
    def run(self):
        raise NotImplementedError


class CollectiveTaskBase(TaskBase):
    def __init__(self, name: str):
        super().__init__(name=name)
    
    # @abstractmethod
    def collect_config(self, task: "TaskBase"):
        if isinstance(self.config_template, dict):
            self.config_template = [self.config_template]
        self.config_template.append(task.config_template)


class UpdateTaskBase(TaskBase):
    def __init__(self, name: str):
        super().__init__(name=name)

    @abstractmethod
    def init_from(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def collect_evidence(self, task: "UpdateTaskBase", **kwargs):
        raise NotImplementedError

    @abstractmethod
    def update_belief(self, **kwargs):
        raise NotImplementedError


class DemoUpdateTask(TaskBase):
    """
    Return
    [
        {
            cluster_id: "dominated"/"possessed",
            "output_ids": t.List
        }
    ]
    
    TODO:
        save binding pose for the core structure as a reference
    """
    def __init__(self, 
                 name: str = 'count_votes'):
        self.name = name
        self.config_template = {'ligand': UNDEFINED_PARAMETER,
                                'poses': UNDEFINED_PARAMETER,
                                'cluster': UNDEFINED_PARAMETER,
                                'dominate_cutoff': UNDEFINED_PARAMETER}
        
        self.collected_votes = defaultdict(Counter)
        self.output_ids = defaultdict(list)
        
    def run(self, other_task, index):
        _cluster = other_task.config_template['cluster']
        _ligand = other_task.config_template['ligand']
        
        self.collected_votes[_ligand].update([_cluster])
        self.output_ids[f"{_ligand}_{_cluster}"].append(index)
        
    def result(self):
        cluster_state = defaultdict(dict)
        
        for _ligand in self.collected_votes:
            all_vote_num = sum(self.collected_votes[_ligand].values())
            for _cluster in self.collected_votes[_ligand]:
                percentage = self.collected_votes[_cluster][_ligand] / all_vote_num
                
                if percentage > self.config_template['dominate_cutoff']:
                    cluster_state[_cluster] = {
                        "dominate": set(),
                        "possess": set()
                    }
                            
        for _ligand in self.collected_votes:
            all_vote_num = sum(self.collected_votes[_ligand].values())
            for _cluster in cluster_state:
                percentage = self.collected_votes[_cluster][_ligand]/all_vote_num
                if percentage > self.config_template['dominate_cutoff']:
                    cluster_state[_cluster]['dominate'].add(_ligand)
                else:
                    cluster_state[_cluster]['possess'].add(_ligand)
        
        outputs = []
        for _cluster in cluster_state:
            for state in ("dominate", "possess"):
                if not cluster_state[_cluster].has_key(state): continue

                collect_output_ids = []
                for _ligand in cluster_state[_cluster][state]:
                    collect_output_ids += self.output_ids[f"{_ligand}_{_cluster}"]

                outputs.append({
                    _cluster: state,
                    "output_ids": collect_output_ids
                })
                
        return outputs