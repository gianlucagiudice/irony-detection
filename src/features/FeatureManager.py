from src.features.EmotionalFeature import EmotionalFeature
from src.features.PosFeature import PosFeature
from src.features.PragmaticParticlesFeature import PragmaticParticlesFeature
from src.features.TextFeature import TextFeature
from src.features.Tokenizer import Tokenizer

FEATURES = ['pp', 'pos', 'emot']


class FeatureManager:

    def __init__(self, tweets, debug=False):
        # List of tweets
        self.tweets = tweets
        # Text feature file
        self.text_feature = None
        # Final matrix
        self.feature_matrix = dict()
        # Debug mode
        self.debug = debug

    def extract_features(self):
        # Evaluate all features
        self.evaluate_features(self.tweets)
        # Return final matrix
        return self.text_feature, self.feature_matrix

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
        self.text_feature = text_features.extract_terms_matrix()

        # ------ Features ------
        # Pragmatic particles
        self.feature_matrix['pp'] = self.compute_pragmatic_particles(tweets)
        # Part of speech features
        self.feature_matrix['pos'] = self.compute_pos_tag(tweets)
        # Emotional features
        self.feature_matrix['emot'] = self.compute_emotional_feature(tokenizer.words_list)

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