from multiprocessing import Pool

from .TextFeature import TextFeature


class Bow(TextFeature):
	name = 'bow'

	chunk_size = 3000
	pool = Pool

	def __init__(self, tweets):
		super().__init__()
		# List of words in each tweet
		self.words_list = tweets.tokens
		# Lis of unique words
		self.unique_words_list = tweets.unique_words_list

	def extract_text_matrix(self):
		super().extract_text_matrix()
		# Fill matrix
		self.fill_matrix(self.compute_row, self.words_list, self.pool, self.chunk_size)
		# Return matrix
		return self.matrix, self.unique_words_list, self.name

	def compute_row(self, words):
		tweet_words_set = set(words)
		return [int(unique_word in tweet_words_set) for unique_word in self.unique_words_list]
