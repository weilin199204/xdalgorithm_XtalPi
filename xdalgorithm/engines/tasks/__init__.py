"""
The tasks of the event sourcing experiment.
Your can write your own tasks to use in the `Dataset` which should return
a `list` of `dict` in the `run()` method. 
The nessesary kwargs should be in `config_template` while adding 
this task to a `Dataset` instance.
"""
from .preparations import (
    AddLigands,
    ProteinFixer,
    ProteinSystemBuilder,
    LigandProcessor,
    LigandConformationGenerator,
    ScaffoldNetwork,
    Core,
    GenericCore,
    RGroupIFPLabels
)
from .docking import (
    AutoGrid,
    AutoDock
)
from .ifp import (
    IFP,
    RGroupIFP
)
from .substructure_matching import (
    Filtering
)

from .md import MDSystemBuilder
from .clustering import (
    InterMoleculeRMSDClustering,
    IntraMoleculeDihedralClustering,
    IntraMoleculeRMSDClustering,
    ProteinDihedralClustering
)

from .aff import (
    QSAR,
    PredictAff
)
from .reports import (
    DockingReports
)


__all__ = [
    'AddLigands',
    'ProteinFixer',
    'ProteinSystemBuilder',
    'LigandProcessor',
    'LigandConformationGenerator',
    'ScaffoldNetwork',
    'Core',
    'GenericCore',
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
    'Filtering',
    'Merge'
]
