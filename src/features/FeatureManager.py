import numpy as np

from src.features.EmotionalFeature import EmotionalFeature
from src.features.PosFeature import PosFeature
from src.features.PragmaticParticlesFeature import PragmaticParticlesFeature
from src.features.TextFeature import TextFeature
from src.features.Tokenizer import Tokenizer


class FeatureManager:

    def __init__(self, tweets, debug=False):
        # List of tweets
        self.tweets = tweets
        # Text feature file
        self.text_feature = None
        # List of matrices
        self.matrices = []
        # Final matrix
        self.matrix = None
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
        text_features = TextFeature(tokenizer.wordsList)
        # Get terms matrix
        self.text_feature = text_features.extract_terms_matrix(debug=self.debug)

        # ------ Pragmatic particles ------
        print("> Pragmatic particles features . . .")
        # Crete pragmatic particles object
        pragmatic_particles = PragmaticParticlesFeature(tweets)
        # Evaluate pragmatic particles for tweets
        self.matrices.append(pragmatic_particles.evaluate_pragmatic_particles(debug=self.debug))

        # ------ Part of speech features ------
        print("> Pos tagging . . .")
        # Create pos tagger object
        part_of_speech = PosFeature(tweets)
        # Tag part of speech
        self.matrices.append(part_of_speech.compute_pos_tags(debug=self.debug))

        #  ------ Emotional features ------
        print("> Emotional feature . . .")
        # Create emotional feature object
        emotional = EmotionalFeature(tokenizer.wordsList)
        # Evaluate emotional feature
        self.matrices.append(emotional.evaluate_emotions(debug=self.debug))

    def build_matrix(self, matrices):
        print("> Building matrix . . .")
        self.matrix = np.concatenate(matrices, axis=1)
