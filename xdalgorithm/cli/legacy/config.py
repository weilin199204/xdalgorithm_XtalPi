import os
import logging

LOG = logging.getLogger(__name__)

class CLICommand:
    """A tool to manage configurations.
    """

    @staticmethod
    def add_arguments(parser):
        # parser.add_argument('--new', dest='init_config', action='store_true')
        pass

    @staticmethod
    def run(args):
        from xdalgorithm.config import init_default_config
        # if args.init_config:
        init_default_config()
        LOG.info('config has been successfully created.')
