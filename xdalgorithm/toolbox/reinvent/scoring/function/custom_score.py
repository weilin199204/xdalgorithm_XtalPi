from os import path as osp
import pathlib
import sys
import importlib
import typing as t


def get_func_from_dir(score_dir: str) -> t.Tuple[t.Callable, str]:
    if score_dir.endswith('.py'):
        func_dir = osp.abspath(pathlib.Path(score_dir).parent.resolve())
        file_name = pathlib.Path(score_dir).stem
    else:
        func_dir = osp.abspath(score_dir)
        file_name = "main"

    sys.path.append(func_dir)
    module = importlib.import_module(file_name) 
    try:
        mode = module.MODE
    except Exception as _:
        mode = 'batch'
    return module.main, mode 