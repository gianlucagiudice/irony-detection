import numpy as np

from src.features.Debugger import Debugger
from src.features.Feature import Feature
from src.lexicons.Lexicon import Lexicon
from src.lexicons.parser.EmoLexParser import EmoLexParser
from src.lexicons.parser.EmoSenticNetParser import EmoSenticNetParser
from src.utils.config import EMOLEX_PATH, EMOSENTICNET_PATH

# Lexicons list
LEXICONS_LIST = [Lexicon(EmoLexParser(EMOLEX_PATH)),
				 Lexicon(EmoSenticNetParser(EMOSENTICNET_PATH))]

# Set of emotions to exclude
EXCLUDE_EMOTIONS_SET = {"positive", "negative"}


class EmotionalFeature(Feature, Debugger):

	def __init__(self, words_list):
		super().__init__()
		self.words_list = words_list
		# Set list of lexicons for emotional features
		self.lexicons = LEXICONS_LIST
		# List of emoticons
		self.emotional_list = []
		# List of emotional feature combined foreach lexicon
		self.emotional_feature_list = []

	def evaluate_emotions(self, debug=False):
		# Evaluate tweets using all lexicons
		self.evaluate_lexicons()
		# Combine lexicons
		self.combine_lexicons()
		# Build matrix
		self.build_matrix(len(self.words_list), len(self.emotional_feature_list[0]))
		# Fill matrix with lexicons value
		self.fill_matrix()
		# Debug info
		self.print_debug_info(debug)
		# Return matrix
		return self.matrix

	def evaluate_lexicons(self):
		for words in self.words_list:
			row = dict()
			for lexicon in self.lexicons:
				# List of values relative to all words
				values = [list(lexicon.evaluate_word(word).values()) for word in words]
				# Sum all values to obtain per-lexicon values
				total = np.array(values).sum(axis=0)
				# Reconstruct dictionary form matrix
				if values:
					row[lexicon] = {e: int(v) for e, v in zip(lexicon.emotions(), total)}
				else:
					row[lexicon] = {e: 0 for e in lexicon.emotions()}
			# Add target lexicon to specific list of words
			self.emotional_list.append(row)

	def combine_lexicons(self):
		all_emotions = set([e for lexicon in self.lexicons for e in lexicon.emotions()])
		# Compute emotions to combine
		target_emotions = all_emotions - EXCLUDE_EMOTIONS_SET
		keys = list(target_emotions)
		# Iterate over each tweet in order to combine each lexicon
		for lexicons in self.emotional_list:
			# Initialize combined dictionary to 0s
			combined = {k: 0 for k in keys}
			# Iterate and combine each lexicon
			for lexicon in lexicons.values():
				for k, v in lexicon.items():
					if k in target_emotions:
						combined[k] = combined[k] + v
			# Append the new combined lexicon
			self.emotional_feature_list.append(combined)

	def fill_matrix(self):
		for r, lexicon in enumerate(self.emotional_feature_list):
			self.matrix[r:] = list(lexicon.values())

	def __str__(self, **kwargs):
		title = "emotional feature"
		header = "Tweet"
		template = "Words\t\t >>> {}\n" + \
				   "{:<13}>>> {}\n" * len(self.lexicons) + \
				   "Combined\t >>> {}"
		# Build auxiliary list in order to print per-lexicon debug info
		aux_debug: [list] = []
		for lexicon in self.lexicons:
			aux_debug += [[str(lexicon)] * len(self.words_list)] + \
						 [[tweet[lexicon] for tweet in self.emotional_list]]
		# Return debug info
		return super().__str__(self, self.words_list, *aux_debug, self.emotional_feature_list,
							   title=title, header=header, template=template)
