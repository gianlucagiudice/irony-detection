from src.dataset.Tweets import Tweets
from src.features.EmotionalFeature import EmotionalFeature
from src.features.PosFeature import PosFeature
from src.features.PragmaticParticlesFeature import PragmaticParticlesFeature
from src.features.text.Tokenizer import Tokenizer


class FeatureManager:

	def __init__(self, tweets, text_feature_strategy, debug=False):
		# List of tweets
		self.tweets = Tweets(tweets)
		# Final matrix
		self.feature_matrix = {'text': None, 'pp': None, 'pos': None, 'emot': None}
		# Text feature strategy
		self.text_feature_strategy = text_feature_strategy
		# Debug mode
		self.debug = debug

	def extract_features(self):
		# Evaluate all features
		self.evaluate_features()
		# Return final matrix
		return self.feature_matrix

	def evaluate_features(self):
		# ------ Tokenizer ------
		tokenizer = Tokenizer(self.tweets.values)
		# Compute all tweets
		self.tweets.tokens, self.tweets.unique_words_list = tokenizer.tokenize()

		# ------ Features ------
		# Text feature
		text_features = self.text_feature_strategy(self.tweets)
		self.feature_matrix['text'] = text_features.extract_text_matrix()
		# Pragmatic particles
		self.feature_matrix['pp'] = self.compute_pragmatic_particles(self.tweets.values)
		# Part of speech features
		self.feature_matrix['pos'] = self.compute_pos_tag(self.tweets.values)
		# Emotional features
		self.feature_matrix['emot'] = self.compute_emotional_feature(self.tweets.tokens)

	def compute_emotional_feature(self, words_list):
		print("> Emotional feature . . .")
		# Create emotional feature object
		emotional = EmotionalFeature(words_list)
		# Evaluate emotional feature
		return emotional.evaluate_emotions(debug=self.debug)

	def compute_pos_tag(self, tweets):
		print("> Pos tagging . . .")
		# Create pos tagger object
		part_of_speech = PosFeature(tweets)
		# Tag part of speech
		return part_of_speech.compute_pos_tags(debug=self.debug)

	def compute_pragmatic_particles(self, tweets):
		print("> Pragmatic particles features . . .")
		# Crete pragmatic particles object
		pragmatic_particles = PragmaticParticlesFeature(tweets)
		# Evaluate pragmatic particles for tweets
		return pragmatic_particles.evaluate_pragmatic_particles(debug=self.debug)
