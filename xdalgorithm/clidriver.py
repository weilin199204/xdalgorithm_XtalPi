import argparse
import textwrap
import logging
from importlib import import_module

from xdalgorithm import __version__

LOG = logging.getLogger('xdalgorithm.clidriver')
LOG_FORMAT = (
    '%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')

available_commands = [
    ('molgen', 'xdalgorithm.cli.legacy.molgen'),
    ('xreact', 'xdalgorithm.cli.legacy.xreact')
]

logo_description = r"""

    __  ______  ____
    \ \/ /  _ \|  _ \
     \  /| | | | | | |
     /  \| |_| | |_| |
    /_/\_\____/|____/

    XtalPi Drug Design Toolkit

"""

def main():
    driver = create_clidriver()
    rc = driver.main()
    return rc

def create_clidriver():
    #TODO: init session if necessary
    #TODO: load plugins
    driver = CLIDirver()
    return driver

class CLIDirver(object):

    def __init__(self):
        self._command_table = None
        self._argument_table = None
    
    def main(self, prog='xdd', description=logo_description,
        version=__version__, commands=available_commands, args=None):
        """

        :param args: List of argumetns, with the 'aws' removed. For exmaple,
            the command "aws s3 list-objects --bucket foo" will have an
            args list of ``['s3', 'list-objects', '--bucket', 'foo']``.

        """
        parser = argparse.ArgumentParser(prog=prog,
                                     description=description,
                                     formatter_class=Formatter)
        parser.add_argument('--version', action='version',
                            version='%(prog)s-{}'.format(version))
        parser.add_argument('-T', '--tracebacok', action='store_true')
        subparsers = parser.add_subparsers(title='Sub-commands',
                                        dest='command')

        subparser = subparsers.add_parser('help',
                                        description='Help',
                                        help='Help for sub-command.')
        subparser.add_argument('helpcommand',
                            nargs='?',
                            metavar='sub-command',
                            help='Provide help for sub-command.')

        functions = {}
        parsers = {}
        # show help information
        for command, module_name in commands:
            cmd = import_module(module_name).CLICommand  # import MMP,Reinvent,
            docstring = cmd.__doc__
            if docstring is None:
                # Backwards compatibility with GPAW
                short = cmd.short_description
                long = getattr(cmd, 'description', short)
            else:
                parts = docstring.split('\n', 1)
                if len(parts) == 1:
                    short = docstring
                    long = docstring
                else:
                    short, body = parts
                    long = short + '\n' + textwrap.dedent(body)
            subparser = subparsers.add_parser(
                command,
                formatter_class=Formatter,
                help=short,
                description=long)
            cmd.add_arguments(subparser) # receive a parser object
            functions[command] = cmd.run   # dictionary:module_name->module_run function
            parsers[command] = subparser


        args = parser.parse_args(args)  # Namespace
        if args.command == 'help':
            if args.helpcommand is None:
                parser.print_help()
            else:
                parsers[args.helpcommand].print_help()
        elif args.command is None:
            parser.print_usage()
        else:
            f = functions[args.command]  # select function according to command
            # try:
            if f.__code__.co_argcount == 1:
                f(args)  # args is a Namespace
            else:
                f(args, parsers[args.command])
            # except KeyboardInterrupt:
            #    pass
            # except Exception as x:
            #    if args.traceback:
            #        raise
            #    else:
            #        l1 = '{}: {}\n'.format(x.__class__.__name__, x)
            #        l2 = ('To get a full traceback, use: {} -T {} ...'
            #         .format(prog, args.command))
            #    parser.error(l1 + l2)
    
    def _show_error(self, msg):
        import sys
        LOG.debug(msg, exc_info=True)
        sys.stderr.write(msg)
        sys.stderr.write('\n')

class Formatter(argparse.HelpFormatter):
    """Improved help formatter."""

    def _fill_text(self, text, width, indent):
        assert indent == ''
        out = ''
        blocks = text.split('\n\n')
        for block in blocks:
            if len(block) > 0:
                if block[0] == '*':
                    # List items:
                    for item in block[2:].split('\n* '):
                        out += textwrap.fill(item,
                                            width=width - 2,
                                            initial_indent='* ',
                                            subsequent_indent='  ') + '\n'
                elif block[0] == ' ':
                    # Indented literal block:
                    out += block + '\n'
                else:
                    # Block of text:
                    # out += textwrap.fill(block, width=width) + '\n'
                    out += block + '\n'

            out += '\n'
        return out[:-1]
