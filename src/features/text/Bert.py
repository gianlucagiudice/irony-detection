from multiprocessing.pool import ThreadPool

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

from .TextFeature import TextFeature


class Bert(TextFeature):

    name = 'bert'

    chunk_size = 1000
    pool = ThreadPool

    def __init__(self, tweets):
        super().__init__()
        # Tweets
        self.tweet_list = tweets.values
        # Tokenizer
        self.tokenizer = None
        # Model
        self.model = None
        # Bert
        self.tokens_list = []
        self.indices_list = []
        self.segments_ids_list = []

    def extract_text_matrix(self):
        super().extract_text_matrix()
        # Create tokenizer
        self.build_tokenizer()
        # Build model
        self.build_model()
        # Fill matrix
        self.fill_matrix(self.compute_row, self.tweet_list, self.pool, self.chunk_size)
        # Return matrix
        return self.matrix, list(range(1, len(self.tweet_list) + 1)), self.name

    def build_tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def build_model(self):
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()

    def tokenize(self, tweet):
        marked_text = '{}{}{}'.format('[CLS] ', tweet, ' [SEP]')
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        return indexed_tokens, segments_ids

    def compute_row(self, tweet):
        # Tokenize tweet
        indexed_tokens, segments_ids = self.tokenize(tweet)
        # Create tensor
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensor = torch.tensor([segments_ids])
        # Predict hidden states
        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, segments_tensor)
        # Average last layer
        tokens_vects = encoded_layers[11][0]
        sentence_embedding = torch.mean(tokens_vects, dim=0)
        # Return tokens average
        return [tensor.item() for tensor in sentence_embedding]
