from multiprocessing.pool import ThreadPool

from sentence_transformers import SentenceTransformer

from .TextFeature import TextFeature


class Sbert(TextFeature):

    name = 'sbert'

    chunk_size = 1000
    pool = ThreadPool

    def __init__(self, tweets):
        super().__init__()
        # Tweets
        self.tweet_list = tweets.values
        # Model
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
        return self.model.encode([tweet])[0]
