import numpy as np

from src.features.EmotionalFeature import EmotionalFeature
from src.features.PosFeature import PosFeature
from src.features.PragmaticParticlesFeature import PragmaticParticlesFeature
from src.features.TextFeature import TextFeature
from src.features.Tokenizer import Tokenizer


class FeatureManager:

    def __init__(self, tweets, target_feature, debug=False):
        # List of tweets
        self.tweets = tweets
        # Text feature file
        self.text_feature = None
        # List of matrices
        self.matrices = []
        # Final matrix
        self.matrix = None
        # Target features
        self.target_features = target_feature
        # Debug mode
        self.debug = debug

    def extract_features(self):
        # Evaluate all features
        self.evaluate_features(self.tweets)
        # Build matrix
        self.build_matrix(self.matrices)
        # Return final matrix
        return self.text_feature, self.matrix

    def evaluate_features(self, tweets):
        # ------ Tokenizer ------
        print("> Tokenizer . . .")
        # Create tokenizer object
        tokenizer = Tokenizer(tweets)
        # Compute all tweets
        tokenizer.parse_tweets(debug=self.debug)

        # ------ Compute text features ------
        print("> Text features . . .")
        # Crete text features object
        text_features = TextFeature(tokenizer.words_list)
        # Get terms matrix
        self.text_feature = text_features.extract_terms_matrix(debug=self.debug)

        # ------ Pragmatic particles ------
        if self.target_features['pp'] is True:
            self.matrices.append(self.compute_pragmatic_particles(tweets))

        # ------ Part of speech features ------
        if self.target_features['pos'] is True:
            self.matrices.append(self.compute_pos_tag(tweets))

        #  ------ Emotional features ------
        if self.target_features['emot'] is True:
            self.matrices.append(self.compute_emotional_feature(tokenizer.words_list))

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

    def build_matrix(self, matrices):
        print("> Building matrix . . .")
        self.matrix = np.concatenate(matrices, axis=1)
