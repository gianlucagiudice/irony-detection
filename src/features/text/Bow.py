from multiprocessing import Pool

from .TextFeature import TextFeature


class Bow(TextFeature):

    name = 'bow'

    chunk_size = 2000
    pool = Pool

    def __init__(self, tweets):
        super().__init__()
        # List of words in each tweet
        self.words_list = tweets.tokens
        # Set of words
        self.words_set = None
        # Set sorted
        self.unique_words_list = None

    def extract_text_matrix(self):
        super().extract_text_matrix()
        # Create set of words
        self.create_words_set()
        # Fill matrix
        self.fill_matrix(self.compute_row, self.words_list, self.pool, self.chunk_size)
        # Return matrix
        return self.matrix, self.unique_words_list, self.name

    def create_words_set(self):
        # Create set of words
        self.words_set = set([word for words in self.words_list for word in words])
        # Build list of unique words
        self.unique_words_list = sorted(list(self.words_set))
        # Print number of words added to dictionary
        print('\t{} words in dictionary'.format(len(self.unique_words_list)))

    def compute_row(self, words):
        tweet_words_set = set(words)
        return [int(unique_word in tweet_words_set) for unique_word in self.unique_words_list]
