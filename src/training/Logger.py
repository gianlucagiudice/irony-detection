import logging
import time

from main import TARGET_DATASET
from src.config import REPORTS_PATH

LOG_FILENAME = 'training'
LOG_FILE_PATH = '{}{}/{}.log'.format(REPORTS_PATH, TARGET_DATASET, LOG_FILENAME)


class Logger:
    __instance = None

    @staticmethod
    def getLogger():
        """ Static access method. """
        if not Logger.__instance:
            Logger()
        return Logger.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if Logger.__instance:
            raise Exception("This class is a singleton!")
        else:
            Logger.__instance = self
            # Create logger
            format = '{} %(asctime)s <%(processName)s> {}\n\t%(message)s\n'\
                .format('=' * 4, '=' * 4)
            logging.basicConfig(filename=LOG_FILE_PATH, filemode='w+', level=logging.INFO,
                                format=format,
                                datefmt='%d-%m-%Y %H:%M:%S',)
            self.logger = logging.getLogger()
            # Start time
            self.start_time = time.time()

    def print(self, string):
        self.logger.info(string.strip())
        print(string)

    def completed(self):
        elapsed_time = time.time() - self.start_time
        h_readable_elapsed = time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed_time))
        elapsed_days = int(elapsed_time // 86400)
        self.print('Execution time: {}d{}'.format(elapsed_days, h_readable_elapsed))
