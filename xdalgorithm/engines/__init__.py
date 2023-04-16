"""
This is the Event-sourcing task engine, you can define your `Dataset` and task
flow here by write a simple sub-class of `TaskBase` which should have a `name` 
and `config_template`.
The `Dataset` holds your data with a `nx.Graph` and you can track your task history
int the event sourcing framework. The examples are as follows.

>>> from xdalgorithm.engines import get_dataset
... from xdalgorithm.engines import AddLigandInfo
... from xdalgorithm.engines import SerialEventBase

>>> data=get_dataset()

>>> data.run_task(
...     SerialEventBase, 
...     task=AddLigandInfo(), 
...     csv_file='datasets/Miransertib_AKT.csv',
...     event_type='LIGAND'
... )
"""
from xdalgorithm.engines.data import (
    get_dataset,
)
from xdalgorithm.engines.events import (
    SerialEventBase,
    ParallelEventBase,
    CollectiveEventBase
)
from xdalgorithm.engines.tasks import (
    AddLigands,
    ProteinFixer,
    ProteinSystemBuilder,
    LigandProcessor,
    LigandConformationGenerator,
    ScaffoldNetwork,
    Core,
    GenericCore,
    RGroupIFPLabels,
    AutoGrid,
    AutoDock,
    IFP,
    RGroupIFP,
    IntraMoleculeDihedralClustering,
    IntraMoleculeRMSDClustering,
    InterMoleculeRMSDClustering,
    ProteinDihedralClustering,
    MDSystemBuilder,
    Filtering,
    QSAR,
    PredictAff,
    DockingReports
)

from xdalgorithm.engines.utils import (
    load_data,
    ch_datadir,
    viz_dataset,
    HashableDict,
)


__all__ = [
    'SerialEventBase',
    'CollectiveEventBase',
    'ParallelEventBase',
    'ProteinFixer',
    'ProteinSystemBuilder',
    'AddLigands',
    'LigandProcessor',
    'LigandConformationGenerator',
    'ScaffoldNetwork',
    'Core',
    'RGroupIFPLabels',
    'AutoGrid',
    'AutoDock',
    'IFP',
    'RGroupIFP',
    'IntraMoleculeDihedralClustering',
    'IntraMoleculeRMSDClustering',
    'InterMoleculeRMSDClustering',
    'ProteinDihedralClustering',
    'MDSystemBuilder',
    'QSAR',
    'PredictAff',
    'DockingReports',
    'Merge',
    'Filtering',
    'HashableDict',
    'load_data',
    'get_dataset',
    'ch_datadir',
    'viz_dataset'
]