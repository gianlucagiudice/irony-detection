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
        self.textFeature = None
        # List of matrices
        self.matrices = []
        # Final matrix
        self.matrix = None
        # Debug mode
        self.debug = debug

    def extractFeatures(self):
        # Evaluate all features
        self.evaluateFeatures(self.tweets)
        # Build matrix
        self.buildMatrix(self.matrices)
        # Return final matrix
        return self.textFeature, self.matrix

    def evaluateFeatures(self, tweets):
        # ------ Tokenizer ------
        print("> Tokenizer . . .")
        # Create tokenizer object
        tokenizer = Tokenizer(tweets)
        # Compute all tweets
        tokenizer.parseTweets(debug=self.debug)

        # ------ Compute text features ------
        print("> Text features . . .")
        # Crete text features object
        text_features = TextFeature(tokenizer.wordsList)
        # Get terms matrix
        self.textFeature = text_features.extractTermsMatrix(debug=self.debug)

        # ------ Pragmatic particles ------
        print("> Pragmatic particles features . . .")
        # Crete pragmatic particles object
        pragmatic_particles = PragmaticParticlesFeature(tweets)
        # Evaluate pragmatic particles for tweets
        self.matrices.append(pragmatic_particles.evaluatePragmaticParticles(debug=self.debug))

        # ------ Part of speech features ------
        print("> Pos tagging . . .")
        # Create pos tagger object
        part_of_speech = PosFeature(tweets)
        # Tag part of speech
        self.matrices.append(part_of_speech.computePosTags(debug=self.debug))

        #  ------ Emotional features ------
        print("> Emotional feature . . .")
        # Create emotional feature object
        emotional = EmotionalFeature(tokenizer.wordsList)
        # Evaluate emotional feature
        self.matrices.append(emotional.evaluateEmotions(debug=self.debug))

    def buildMatrix(self, matrices):
        print("> Building matrix . . .")
        self.matrix = np.concatenate(matrices, axis=1)
