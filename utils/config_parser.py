import os
from utils import custom_logging
from configparser import ConfigParser


class ConfParser(object):
    """
    Configuration file parser
    """
    def __init__(self):
        dir_path = os.path.dirname(__file__)
        self._config_file_path = os.path.join(dir_path, '../config.cfg')
        self.parser = self._config_reader()

    def _config_reader(self):
        parser = ConfigParser()
        try:
            with open(self._config_file_path) as fp:
                parser.read_file(fp)
        except FileNotFoundError:
            custom_logging.critical('Config file not found')
            exit()

        return parser
