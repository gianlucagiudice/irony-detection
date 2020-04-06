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
        self.wordsList = words_list
        self.wordsSet = None
        self.uniqueWordsList = None

    def extractTermsMatrix(self, debug=False):
        # Create set of words
        self.createWordsSet()
        # Fill matrix
        self.fillMatrix()

        # TODO: Rremove
        print(len(self.uniqueWordsList))
        with open('words.list', 'w') as out:
            for x in self.uniqueWordsList:
                out.write(x + '\n')
        #quit()

        # Return matrix
        return self.matrix, self.uniqueWordsList

    def createWordsSet(self):
        # Create set of words
        self.wordsSet = set([word for words in self.wordsList for word in words])
        # Build list of unique words
        self.uniqueWordsList = sorted(list(self.wordsSet))

    def fillMatrix(self):
        # Create matrix file
        matrix_file = tempfile.NamedTemporaryFile(mode='w+')
        # Compute all chunks
        print('\t{}% completed'.format(0), end='')
        for index, chunk in enumerate(self.chunkWords()):
            with Pool(THREAD_NUMBER) as pool:
                computed = pool.map(self.computeRow, chunk)
            self.writeChunk(matrix_file, computed)
            print('\r\t{}% completed'.format(round((index+1)*CHUNK_SIZE/len(self.wordsList)*100, 3)), end='')
        print('\r\t100% processed')
        # Save matrix file
        self.matrix = matrix_file

    def computeRow(self, words):
        tweet_words_set = set(words)
        return [int(unique_word in tweet_words_set) for unique_word in self.uniqueWordsList]

    def writeChunk(self, file, chunk):
        with open(file.name, mode='a+') as csvfile:
            writer = csv.writer(csvfile)
            for row in chunk:
                writer.writerow(row)

    def chunkWords(self):
        for i in range(0, len(self.wordsList), CHUNK_SIZE):
            yield self.wordsList[i:i + CHUNK_SIZE]

    def __str__(self):
        return ''
    # def __str__(self, **kwargs):
    #     title = "text features"
    #     header = "Matrix row"
    #     template = "Uniq. words >>> {}\n" \
    #                "Tweet words\t>>> {}\n" \
    #                "Row\t\t\t>>> {}"
    #     return super().__str__(self, [self.uniqueWordsList] * len(self.matrix), self.wordsList, self.matrix,
    #                            title=title, header=header, template=template)
