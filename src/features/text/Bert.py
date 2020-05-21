import re
from multiprocessing.pool import ThreadPool

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

from .TextFeature import TextFeature

valid_token_regexp = "#*[a-zA-Z]+"
exclude_words_set = {'irony', 'ironic', 'rt'}   # RT = retweet


class Bert(TextFeature):

    name = 'bert'

    chunk_size = 1000
    pool = ThreadPool


    def __init__(self, tweets):
        super().__init__()
        # Tweets
        self.tweet_list = tweets.values
        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Model
        self.model = Bert.build_model()
        # Bert
        self.tokens_list = []
        self.indices_list = []
        self.segments_ids_list = []
        self.feature_length = 0

    def extract_text_matrix(self):
        super().extract_text_matrix()
        # Fill matrix
        self.feature_length = self.fill_matrix(self.compute_row, self.tweet_list, self.pool, self.chunk_size)
        # Return matrix
        return self.matrix, list(range(1, self.feature_length + 1)), self.name

    def compute_row(self, tweet):
        # Tokenize tweet
        indexed_tokens, segments_ids = self.tokenize(tweet, self.tokenizer)
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

    @staticmethod
    def build_model():
        model = BertModel.from_pretrained('bert-base-uncased')
        model.eval()
        return model

    @staticmethod
    def tokenize(tweet, tokenizer, exclude_hot_words=True):
        # Mark tweet
        marked_text = '{}{}{}'.format('[CLS] ', tweet, ' [SEP]')
        # Tokenize tweet
        tokenized_text = tokenizer.tokenize(marked_text)
        # Exclude invalid tokens
        tokenized_text_valid = Bert.validate_tokens(tokenized_text, exclude_hot_words)
        # Convert tokens to Indexes
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text_valid)
        segments_ids = [1] * len(tokenized_text_valid)
        return indexed_tokens, segments_ids

    @staticmethod
    def validate_tokens(tokenized_text, exclude_hot_words):
        tokenized_text_valid = [token for token in tokenized_text[1:-1] if re.fullmatch(valid_token_regexp, token)]
        if exclude_hot_words:
            tokenized_text_valid = [token for token in tokenized_text[1:-1] if token not in exclude_words_set]
        return [tokenized_text[0]] + tokenized_text_valid + [tokenized_text[-1]]
