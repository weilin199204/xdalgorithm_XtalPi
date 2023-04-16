from .base import CollectiveTaskBase


class Merge(CollectiveTaskBase):
    """ Merge nodes (from diffrent tasks) to the same task output.
        Args:
            name (str,optional): the task name. Default to 'merge'

        Examples:
        >>> data.merge_nodes(['5.TASK.autodock_1', '5.TASK.autodock_2'])
        """
    def __init__(self, name: str = 'merge'):
        super().__init__(name)
        self.config_template = {}

    def run(self):
        for source_id, each_node_dict in enumerate(self.config_template):
            each_node_dict.update({'output_ids': [source_id]})

        return self.config_template