import csv
import tempfile
from multiprocessing import Pool

from src.Config import THREAD_NUMBER
from src.features.Debugger import Debugger
from src.features.Feature import Feature

CHUNK_SIZE = 2000


class TextFeature(Feature, Debugger):

    def __init__(self, words_list):
        super().__init__()
        self.words_list = words_list
        self.words_set = None
        self.unique_words_list = None

    def extract_terms_matrix(self):
        # Create set of words
        self.create_words_set()
        # Fill matrix
        self.fill_matrix()
        # Return matrix
        return self.matrix, self.unique_words_list

    def create_words_set(self):
        # Create set of words
        self.words_set = set([word for words in self.words_list for word in words])
        # Build list of unique words
        self.unique_words_list = sorted(list(self.words_set))
        # Print number of words added to dictionary
        print('\t{} words in dictionary'.format(len(self.unique_words_list)))

    def save_words_list(self):
        with open('words.list', 'w') as out:
            for x in self.unique_words_list:
                out.write(x + '\n')

    def fill_matrix(self):
        # Create matrix file
        matrix_file = tempfile.NamedTemporaryFile(mode='w+')
        # Compute all chunks
        print('\t{}% completed'.format(0), end='')
        for index, chunk in enumerate(self.chunk_words()):
            with Pool(THREAD_NUMBER) as pool:
                computed = pool.map(self.compute_row, chunk)
            self.write_chunk(matrix_file, computed)
            print('\r\t{}% completed'.format(round((index+1) * CHUNK_SIZE / len(self.words_list) * 100, 3)), end='')
        print('\r\t100% processed')
        # Save matrix file
        self.matrix = matrix_file

    def compute_row(self, words):
        tweet_words_set = set(words)
        return [int(unique_word in tweet_words_set) for unique_word in self.unique_words_list]

    def write_chunk(self, file, chunk):
        with open(file.name, mode='a+') as csvfile:
            writer = csv.writer(csvfile)
            for row in chunk:
                writer.writerow(row)

    def chunk_words(self):
        for i in range(0, len(self.words_list), CHUNK_SIZE):
            yield self.words_list[i:i + CHUNK_SIZE]
