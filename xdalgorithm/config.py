import os
import yaml
import shutil
import logging

from xdalgorithm.utils import get_config_file_path

LOG = logging.getLogger(__name__)
CONFIG_FILE = os.path.expanduser('~/.config/xdkit/config.yaml')

def load_or_init_config_if_not_exist(export_to_env=True):
    if os.path.exists(CONFIG_FILE):
        return load_config(CONFIG_FILE, export_to_env=export_to_env)
    else:
        return init_default_config(export_to_env=export_to_env)

def load_default_config(export_to_env=True):
    return load_config(CONFIG_FILE, export_to_env=export_to_env)

# singleton
_config_cache = None
def load_config(config_file, export_to_env=False):
    global _config_cache
    if not _config_cache:
        if os.path.isfile(config_file):
            with open(CONFIG_FILE, 'r') as stream:
                try:
                    _config_cache = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    raise exc
    
    # export as ENV variables
    if export_to_env:
        os.environ['rootPath'] = _config_cache['cli']['ENV_ROOT_PATH'] if 'rootPath' not in os.environ.keys() else os.environ['rootPath']
        os.environ['MGLPY'] = _config_cache['cli']['ENV_MGLPY'] if 'MGLPY' not in os.environ.keys() else os.environ['MGLPY']
        os.environ['MGLUTIL'] = _config_cache['cli']['ENV_MGLUTIL'] if 'MGLUTIL' not in os.environ.keys() else os.environ['MGLUTIL']
        os.environ['JAVA_HOME'] = _config_cache['cli']['ENV_JAVA_HOME'] if 'JAVA_HOME' not in os.environ.keys() else os.environ['JAVA_HOME']
        os.environ['MODULE_HOME'] = _config_cache['cli']['ENV_MODULE_INCLUDE_PATH'] if 'MODULE_HOME' not in os.environ.keys() else os.environ['MODULE_HOME']
        os.environ['KEY_MODELLER'] = _config_cache['cli']['ENV_KEY_MODELLER'] if 'KEY_MODELLER' not in os.environ.keys() else os.environ['KEY_MODELLER']

        #TODO: too much environmet coupling variables here...
        os.environ['PATH'] = os.environ['JAVA_HOME']+'/bin:/' + os.environ['rootPath'] + '/miniconda3/envs/Drug_Design_Backend/bin:' \
                            + os.environ['PATH']+':' + os.environ['MODULE_HOME']+ '/amber20/bin:' + os.environ['MODULE_HOME']+'/vmd/bin:' + \
                            os.environ['MODULE_HOME'] + '/AutoDock:' + os.environ['MODULE_HOME'] + '/mgltools_x86_64Linux2_1.5.6/bin:' + \
                            os.environ['MODULE_HOME']+'/lovoalign-master/bin:' + os.environ['MODULE_HOME']+ '/LeDock:' + \
                            os.environ['MODULE_HOME'] + '/rosetta_src_2020.46.61480_bundle/main/source/bin:' + os.environ['MODULE_HOME']+ '/chemaxon'

        chemaxon_config_file = '{}/.chemaxon'.format(os.environ['rootPath'])
        if os.path.exists(chemaxon_config_file):
            init_chemaxon_config = os.system("cp -rf {0} {1}/".format(
                                    chemaxon_config_file, os.environ['HOME']))
            if init_chemaxon_config != 0:
                LOG.error("failed to execute the command line:", "cp -rf {0} {1}/".format(
                    chemaxon_config_file, os.environ['HOME']), "status is:", init_chemaxon_config)
        else:
            LOG.error("chemaxon config file {} does not exist.".format(chemaxon_config_file))
                
        mysql_config_file = '{}/mysql.conf'.format(os.environ['rootPath'])
        if os.path.exists(mysql_config_file):
            init_mysql_config = os.system("cp -rf {0} {1}/".format(
                                    mysql_config_file, os.environ['HOME']))
            if init_mysql_config != 0:
                LOG.error("failed to execute the command line:", "cp -rf {0} {1}/".format(
                    mysql_config_file, os.environ['HOME']),"status is:",init_mysql_config)
        else:
            LOG.error("mysql config file {} does not exist.".format(mysql_config_file))

    return _config_cache

def init_default_config(export_to_env=True):
    if not os.path.exists(os.path.dirname(CONFIG_FILE)):
        os.makedirs(os.path.dirname(CONFIG_FILE))

    if os.path.isfile(CONFIG_FILE):
        # config file existed, overwrite it
        LOG.warning("config file already exists and will be overwritten.")

    config_file_path = get_config_file_path('default-init.yaml')
    shutil.copyfile(config_file_path, CONFIG_FILE)

    #FIXME: update modeller license
    reset_modeller_license()

    return load_config(CONFIG_FILE, export_to_env=export_to_env)

def reset_modeller_license():
    # get the location of the config.py file
    exc_str = None
    try:
        import modeller
    except ImportError as exc:
        raise Exception(
            "has not installed modeller\nPlease run `conda install -c salilab modeller.")
    except Exception as exc:
        exc_str = str(exc)
    
    if not exc_str:
        return
    
    for l in exc_str.split('\n'):
        if l.endswith('config.py'):
            config_file = l
            # update the license line in the file
            with open(config_file) as f:
                modeller_config = f.read()
            
            has_added_license = False
            new_config = []
            for lc in modeller_config.split('\n'):
                if lc.startswith('license'):
                    new_config.append("license = r'MODELIRANJE'")
                    has_added_license = True
                else:
                    new_config.append(lc)
            if not has_added_license:
                new_config.append("license = r'MODELIRANJE'")
            
            new_config_str = '\n'.join(new_config)
            with open(config_file, 'w') as fw:
                fw.write(new_config_str)
            
            LOG.info('Modeller license in the file {} has been updated'.format(config_file))
            return config_file
    
    LOG.error('cannot update the license of modeller in the config.py.')


