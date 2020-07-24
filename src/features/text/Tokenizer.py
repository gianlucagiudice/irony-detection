import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

from src.features.Debugger import Debugger
from src.utils.config import DATASET_PATH_OUT
from src.utils.parameters import TARGET_DATASET

OCCURRENCE_THRESHOLD = 10


def init_stopwords():
	try:
		return set(stopwords.words('english'))
	except LookupError:
		nltk.download('stopwords')
		return set(stopwords.words('english'))


class Tokenizer(Debugger):
	valid_token_regexp = "[a-zA-Z]+"
	stopwords_set = init_stopwords()
	exclude_words_set = {'irony', 'ironic', 'rt'}  # RT = retweet

	def __init__(self, tweets):
		# List of tweets
		self.tweets_list = tweets
		# List of words in each tweet
		self.words_list = []
		# Words occurrences
		self.word_occurrence = dict()
		# List of unique words
		self.unique_words_list = None
		# Stemmer
		self.stemmer = PorterStemmer()

	def tokenize(self):
		print("> Tokenizer . . .")
		# Extract valid words
		self.extract_words()
		# Remove words below threshold
		self.words_list = self.filter_below_threshold(self.words_list)
		# Create list of unique words
		self.build_unique_words_list()
		# Return list of words
		return self.words_list, self.unique_words_list

	def extract_words(self):
		# Construct tokenizer object
		tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
		# Start tokenizing
		print('\t{}% completed'.format(0), end='')
		# Tokenize all tweets
		for index, tweet in enumerate(self.tweets_list, 1):
			tokens = tokenizer.tokenize(tweet)
			words = [token for token in tokens if self.is_valid_token(token)]
			words_stemmed = self.stem_words(words)
			self.update_occurrences(words_stemmed)
			self.words_list.append(words_stemmed)
			# Progress
			if index % 1000 == 0:
				print('\r\t{}% completed'.format(round(index / len(self.tweets_list) * 100, 3)), end='')
		print('\r\t{}% tokenized'.format(100))

	def filter_below_threshold(self, words_list):
		filtered = []
		for words in words_list:
			filtered.append([word for word in words if self.word_occurrence[word] > OCCURRENCE_THRESHOLD])
		return filtered

	def update_occurrences(self, words):
		for word in words:
			self.word_occurrence[word] = self.word_occurrence.get(word, 0) + 1

	def is_valid_token(self, token):
		return re.fullmatch(self.valid_token_regexp, token)

	def build_unique_words_list(self):
		stopwords_stemmed_set = set(self.stem_words(self.stopwords_set))
		exclude_words_stemmed_set = set(self.stem_words(self.exclude_words_set))
		# Create set of unique words
		words_flatten = [word for words in self.words_list for word in words]
		unique_words_set = set(words_flatten) - stopwords_stemmed_set - exclude_words_stemmed_set
		# Create list to return
		self.unique_words_list = sorted(list(unique_words_set))
		# Write words list to file
		self.unique_words_verbose()

	def stem_words(self, words):
		return [self.stemmer.stem(word) for word in words]

	def __str__(self, **kwargs):
		title = "tokenizer"
		header = "Tweet"
		template = "Original\t>>> \"{}\"\n" \
				   "Valid words\t>>> {}"
		return super().__str__(self, self.tweets_list, self.words_list,
							   title=title, header=header, template=template)

	def unique_words_verbose(self):
		# Print number of words added to dictionary
		print('\t{} Unique words in documents'.format(len(self.unique_words_list)))
		# Write words to file
		filename = '{}{}/unique_words.list'.format(DATASET_PATH_OUT, TARGET_DATASET)
		with open(filename, 'w+') as file:
			file.writelines('\n'.join(self.unique_words_list))
