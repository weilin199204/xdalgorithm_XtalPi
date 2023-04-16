import json
import random
import string
import bson
import typing as t

__all__ = [
    'get_config_file_path',
    'get_template_file_path',
]

import os
from pkg_resources import resource_filename


def get_config_file_path(relative_path):
    """Get the full path to one of the config files in testsystems.

    Parameters
    ----------
    relative_path : str
        Name of the file to load (with respect to the repex folder).
    """
    return get_data_file_path('data/config', relative_path)


def get_template_file_path(relative_path):
    """Get the full path to one of the tempalte files in the testsystems.

    Parameters
    ----------
    relative_path : str
        Name of the tempalte file to load (with respect to the repex folder).
    """
    return get_data_file_path('data/template', relative_path)


def get_data_file_path(subdir, relative_path):
    """Get the full path to one of the files under the xdalgorithm/data folder.

    Parameters
    ----------
    subdir: str
        Name of the subdir under the `xdalgorithm/data` folder.
    relative_path: str
        Name of the file to load (with respect to the repex folder).
    """
    fn = resource_filename(
        'xdalgorithm',
        os.path.join(subdir, relative_path))

    if not os.path.exists(fn):
        raise ValueError("Sorry! {} does not exist.".format(fn))

    return fn


def load_arguments_from_json(json_file) -> t.Dict:
    """Parse arguments in JSON format and return dict objects.

    Parameters
    ----------
    json_file: str
        Path of the input json file.
    """
    with open(json_file) as f:
        json_raw = f.read().replace('\r', '').replace('\n', '')

    # arguments = {}
    try:
        arguments = json.loads(json_raw)
    except ValueError as e:
        raise Exception(
            "Invalid JSON: %s\nJSON received: %s"
            % (e, json_raw))

    return arguments


def save_arguments_to_json(arguments, json_file) -> str:
    """Save arguments into JSON file format.

    Parameters
    ----------
    arguments: dict
        Map of arguments.
    json_file: str
        Path of saving arguments as JSON file.
    """
    json_raw = json.dumps(arguments,
                          default=lambda x: x.__dict__,
                          sort_keys=True,
                          indent=4,
                          separators=(',', ': '))
    with open(json_file, 'w') as f:
        f.write(json_raw)

    return json_file


def get_simple_rand_id(length=8):
    """Return a random and specific length of string

    Parameters
    ----------
    length: str
        the total length of the returned string.
    """
    return ''.join(random.sample(string.ascii_letters + string.digits, length))


def get_rand_id():
    return str(bson.ObjectId())[::-1]

# fs_endpoint: os.environ['XDA_FS_ENDPOINT'] = "http://localhost:9000/test/"
def get_static_uri(fsrc_arr, prefix_dir="/data/aidd-server/hiexplorer/api/public/test", fs_endpoint=os.environ.get('XDA_FS_ENDPOINT', 'http://172.23.50.10:9000/test/')):
    import errno
    import hashlib
    BLOCKSIZE = 65536

    fdst_arr = []

    # hash the content of each source files
    if 0 == len(fsrc_arr):
        raise Exception("input file(s) should not be empty.")

    for index, fsrc in enumerate(fsrc_arr):
        expaneded_fsrc = os.path.abspath(os.path.expanduser(fsrc))
        if not os.path.exists(expaneded_fsrc):
            # input file does not exist
            fdst_arr.append(None)
        else:
            hasher = hashlib.md5()
            with open(expaneded_fsrc, 'rb') as f:
                buf = f.read(BLOCKSIZE)
                while len(buf) > 0:
                    hasher.update(buf)
                    buf = f.read(BLOCKSIZE)
            hashed_fname = hasher.hexdigest()
            symlink_fdst = os.path.join(prefix_dir, hashed_fname)
            # overwrite if file already exists
            try:
                os.symlink(fsrc, symlink_fdst)
            except OSError as e:
                if e.errno == errno.EEXIST:
                    os.remove(symlink_fdst)
                    os.symlink(fsrc, symlink_fdst)

            # fdst_arr.append(symlink_fdst)
            fdst_arr.append(os.path.join(fs_endpoint, hashed_fname))
            # construct static file URI with fs_endpoint

    return fdst_arr