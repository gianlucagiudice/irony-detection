import time

from src.Config import TARGET_DATASET, REPORTS_PATH

import logging

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
        h_readable_elapsed = time.strftime("%dd%Hh%Mm%Ss", time.gmtime(elapsed_time))
        self.print('Execution time: {}'.format(h_readable_elapsed))
