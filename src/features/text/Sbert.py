import re
from multiprocessing.pool import ThreadPool

from sentence_transformers import SentenceTransformer

from .TextFeature import TextFeature


class Sbert(TextFeature):
	name = 'sbert'

	chunk_size = 1000
	pool = ThreadPool

	exclude_words_set = {'irony', 'ironic', 'rt'}

	def __init__(self, tweets):
		super().__init__()
		# Tweets
		self.tweet_list = tweets.values
		# Model
		'''
		Github Repo: https://github.com/UKPLab/sentence-transformers 
		'''
		self.model = SentenceTransformer('bert-base-nli-mean-tokens')
		# Sentence BERT feature length
		self.feature_length = 0

	def extract_text_matrix(self):
		super().extract_text_matrix()
		# Fill matrix
		self.feature_length = self.fill_matrix(self.compute_row, self.tweet_list, self.pool, self.chunk_size)
		# Return matrix
		return self.matrix, list(range(1, self.feature_length + 1)), self.name

	def compute_row(self, tweet):
		target_tweet = self.remove_hot_words(tweet.lower())
		return self.model.encode([target_tweet])[0]

	def remove_hot_words(self, tweet):
		out_tweet = tweet
		for hot_word in self.exclude_words_set:
			# Search for hot_word in tweet
			for match in re.finditer(hot_word, tweet):
				start, end = match.start(), match.end()
				# Replace hot word with placeholder
				out_tweet = out_tweet[:start] + '\0' * (end - start) + out_tweet[end:]
		return re.sub('\0', '', out_tweet)
