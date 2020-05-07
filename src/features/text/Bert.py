import re
from multiprocessing.pool import ThreadPool

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

from .TextFeature import TextFeature


class Bert(TextFeature):

    name = 'bert'

    chunk_size = 1000
    pool = ThreadPool

    valid_token_regexp = "#*[a-zA-Z]+"
    exclude_words_set = {'irony', 'ironic', 'rt'}   # RT = retweet

    def __init__(self, tweets):
        super().__init__()
        # Tweets
        self.tweet_list = tweets.values
        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Model
        self.model = None
        # Bert
        self.tokens_list = []
        self.indices_list = []
        self.segments_ids_list = []
        self.feature_length = 0

    def extract_text_matrix(self):
        super().extract_text_matrix()
        # Build model
        self.build_model()
        # Fill matrix
        self.feature_length = self.fill_matrix(self.compute_row, self.tweet_list, self.pool, self.chunk_size)
        # Return matrix
        return self.matrix, list(range(1, self.feature_length + 1)), self.name

    def build_model(self):
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()

    def tokenize(self, tweet):
        # Mark tweet
        marked_text = '{}{}{}'.format('[CLS] ', tweet, ' [SEP]')
        # Tokenize tweet
        tokenized_text = self.tokenizer.tokenize(marked_text)
        # Exclude invalid tokens
        tokenized_text_valid = self.validate_tokens(tokenized_text)
        # Convert tokens to Indexes
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text_valid)
        segments_ids = [1] * len(tokenized_text_valid)
        return indexed_tokens, segments_ids

    def validate_tokens(self, tokenized_text):
        tokenized_text_valid = [token for token in tokenized_text[1:-1]
                                if re.fullmatch(self.valid_token_regexp, token)
                                and token not in self.exclude_words_set]
        return [tokenized_text[0]] + tokenized_text_valid + [tokenized_text[-1]]

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
