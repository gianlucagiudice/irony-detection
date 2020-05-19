import csv
import tempfile
from abc import ABCMeta, abstractmethod

from src.features.Debugger import Debugger
from src.features.Feature import Feature
from src.utils.config import THREAD_NUMBER


class TextFeature(Feature, Debugger, metaclass=ABCMeta):

    name = "text"

    def __init__(self):
        super().__init__()

    @ abstractmethod
    def extract_text_matrix(self):
        print("> {} features . . .".format(self.name.upper()))

    @ abstractmethod
    def compute_row(self, words):
        pass

    def fill_matrix(self, map_function, target_list, pool_constructor, chunk_size):
        # Create matrix file
        matrix_file = tempfile.NamedTemporaryFile(mode='w+')
        # Compute all chunks
        print('\t{}% completed'.format(0), end='')
        for index, chunk in enumerate(self.chunk_words(target_list, chunk_size), 1):
            with pool_constructor(THREAD_NUMBER) as pool:
                computed = pool.map(map_function, chunk)
            self.write_chunk(matrix_file, computed)
            print('\r\t{}% completed'.format(round(index * chunk_size / len(target_list) * 100, 3)), end='')
        print('\r\t100% processed')
        # Save matrix file
        self.matrix = matrix_file
        # Return features dimensions
        return len(computed[0])

    @staticmethod
    def write_chunk(file, chunk):
        with open(file.name, mode='a+') as csvfile:
            writer = csv.writer(csvfile)
            for row in chunk:
                writer.writerow(row)

    @staticmethod
    def chunk_words(target_list, chunk_size):
        for i in range(0, len(target_list), chunk_size):
            yield target_list[i:i + chunk_size]
