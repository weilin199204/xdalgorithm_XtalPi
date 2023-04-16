'''
EventBase controls how task runs and updates the `Dataset`.
You should prepare a `task` function and pass it to `Dataset.run_task` while passing
the `EventBase` to `Dataset.run_task()`.
'''
import itertools
import traceback

from eventsourcing.domain.model.aggregate import BaseAggregateRoot
# import numpy as np

from .utils import load_data


__all__ = [
    "SerialEventBase",
    "CollectiveEventBase",
    "ParallelEventBase",
]


class ParallelEventBase(BaseAggregateRoot.Event):
    def mutate(self, obj) -> None:
        import multiprocess as mp
        from .utils import collect_results
        manager = mp.Manager()
        # q = manager.Queue()
        pool = mp.Pool(mp.cpu_count())
        configs = manager.list(self.what)
        results = pool.map(collect_results, configs)

        done_list = []
        for _, _, _, _, done in results:
            done_list.append(done)
        
        for outputs, source, event_type, event_name, _, in results:
            for _r in outputs:
                source = source if 'source' not in _r else _r['source']
                obj.update_result(_r, event_type, event_name, source)
        obj.snapshot()


class SerialEventBase(BaseAggregateRoot.Event):
    """
    Define a base events class for MVP1's usage
    
    For developers:
    * fields in what:
        - event_name: str
        - task: t.Callable
        - config_template: t.Dict
        - event_type: str (default: 'TASK')
        - other arguments to update config_template
    * task must have task.name, task.update_config(**kwargs), task.run()
    * `output` of model must be a list(dict)
    * key starting with '_' in the `output` will not be saved
    * config_template contains argument name: argument value for model to run
    if argument value is 'UNDEFINED', it must be defined in self.what

    """

    def mutate(self, obj):
        try:
            for _job in self.what:
                task = _job['task']

                # event_type is TASK by default for running tasks
                event_type = _job['event_type']
                event_name = task.name

                # run model here
                # TODO: add model logger function if needed
                _outputs = task.run()  # t.List[t.Dict]
                
                # task_completed_flags_list.append(task.is_completed)
                for _r in _outputs:
                    # output may specify the source nodes
                    # you can define 'source' in each result or in each task['source']
                    source = _job['source'] if 'source' not in _r else _r['source']
                    # print(source)
                    obj.update_result(_r, event_type, event_name, source)
            obj.snapshot()
        except Exception as e:
            print(e)
            traceback.print_tb(e.__traceback__)
            print(f'ERROR! The task does not finish.')
            if obj.snapshot_idx != -1:
                print('Loading the last snapshot...')
                load_data(obj, obj.snapshot_idx)


class CollectiveEventBase(BaseAggregateRoot.Event):
    """
    CollectiveEventBase will collect all tasks' config as one list and run one single task
    Task in CollectiveEventBase should have a function `collect_config` to define how configs are merged
    """

    def mutate(self, obj):
        try:
            task = self.what[0]['task']
            event_name = task.name
            event_type = self.what[0]['event_type']
            source = [self.what[0]['source']]
            for _job in self.what[1:]:
                task.collect_config(_job['task'])
                source.append(_job['source'])

            _outputs = task.run()

            print('\nTask done! Updating results...')
            for _r in _outputs:
                if 'output_ids' in _r:
                    _source = [source[i] for i in _r['output_ids']]
                    _source = itertools.chain.from_iterable(_source)
                    _source = list(set(_source))
                    del _r['output_ids']
                else:
                    # _source = source[i]
                    _source = list(set(list(itertools.chain.from_iterable(source))))
                obj.update_result(_r, event_type, event_name, _source)
            obj.snapshot()
        except Exception as e:
            print(e)
            traceback.print_tb(e.__traceback__)
            print(f'ERROR! The task dose not finish.')
            if obj.snapshot_idx != -1:
                print('Loading the last snapshot...')
                load_data(obj, obj.snapshot_idx)


class SequentialEventBase(BaseAggregateRoot.Event):
    """
    SequentialEventBase will run tasks sequentially before outputting results
    Task should have method `result()` to output the final results
    """
    def mutate(self, obj):
        try:
            task = self.what[0]['task']

            event_type = self.what[0]['event_type']
            event_name = task.name
            source = []

            for i, _job in enumerate(self.what):
                task.run(_job['task'], i)
                source.append(_job['source'])

            _outputs = task.result()
            print('\nTask done! Updating results...')
            for i, _r in enumerate(_outputs):
                if 'output_ids' in _r:
                    _source = [source[i] for i in _r['output_ids']]
                    _source = itertools.chain.from_iterable(_source)
                    _source = list(set(_source))
                else:
                    # _source = source[i]
                    _source = list(set(list(itertools.chain.from_iterable(source))))
                obj.update_result(_r, event_type, event_name, _source)
            obj.snapshot()
        except Exception as e:
            print(e)
            traceback.print_tb(e.__traceback__)
            print(f'ERROR! The task dose not finish.')
            if obj.snapshot_idx != -1:
                print('Loading the last snapshot...')
                load_data(obj, obj.snapshot_idx)

