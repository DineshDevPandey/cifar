import os
import logging

from utils.config_parser import ConfParser


class Logging(object):
    """
    class to implement custom logging
    """
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self._parser = ConfParser().parser
        self._get_handler()
        self._get_formatter()
        self._get_log_level()
        self._set_logger()

    def _get_log_level(self):
        log_level = self._parser.get('LOGGING', 'LogLevel')

        if log_level == 'INFO':
            self.log_level = logging.INFO
        elif log_level == 'WARNING':
            self.log_level = logging.WARNING
        elif log_level == 'ERROR':
            self.log_level = logging.ERROR
        elif log_level == 'CRITICAL':
            self.log_level = logging.CRITICAL
        else:
            # default level
            self.log_level = logging.DEBUG

    def _get_formatter(self):
        self._formatter = logging.Formatter('%(asctime)s : %(levelname)-8s  : %(message)s')

    def _get_handler(self):
        handler = self._parser.get('LOGGING', 'LogHandler')
        if handler == 'FileHandler':
            log_dir = self._parser.get('LOGGING', 'LogDir')
            if log_dir == '':
                log_dir = '../log'
            else:
                log_dir = '../' + log_dir

            log_file = os.path.join(os.path.dirname(__file__),
                                    os.path.join(log_dir, 'cifar.log'))
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            self._handler = logging.FileHandler(log_file)

        if handler == 'StreamHandler':
            self._handler = logging.StreamHandler()

    def _set_logger(self):
        self._handler.setFormatter(self._formatter)
        self.logger.addHandler(self._handler)
        # self.logger.setLevel(logging.INFO)
        self.logger.setLevel(self.log_level)

    def get_logger(self):
        self.logger.info('*' * 100)
        self.logger.info('Program execution start')
        self.logger.info('*' * 100)
        return self.logger
