import threading
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.dataset.Dataset import Dataset
from src.features.text.Bert import Bert
from src.utils.config import REPORTS_PATH, THREAD_NUMBER
from src.utils.parameters import TARGET_DATASET


class Pca:

    def __init__(self, tweets):
        # Tweets
        self.tweet_list = tweets
        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Model
        self.model = Bert.build_model()
        # Matrix dict
        self.average_dict = dict()
        # Matrix
        self.matrix = None
        # Idx
        self.idx = None
        # Principal component
        self.principal_components = None
        # Out dataframe
        self.df = None
        # Lock
        self.lock = threading.Lock()

    def compute_matrix(self):
        # Compute dict
        self.compute_dict()
        # Write dict to file
        # Create matrix
        self.fill_matrix()
        # Fit pca
        self.transform()
        # Create dataframe
        self.build_df()
        # Save dataframe
        self.dump_df()

    def compute_dict(self):
        with ThreadPool(THREAD_NUMBER) as pool:
            pool.map(self.compute_row, self.tweet_list)
        print("\r\t{}% completed".format(100))

    def compute_row(self, tweet):
        # Tokenize tweet
        indexed_tokens, segments_ids = Bert.tokenize(tweet, self.tokenizer)
        # Create tensor
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensor = torch.tensor([segments_ids])
        # Predict hidden states
        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, segments_tensor)
        token_embeddings = Pca.reshape_layers(encoded_layers)
        # Update dict
        for idx, token in zip(indexed_tokens, token_embeddings):
            cat_token = np.array(Pca.pooling_strategy(token))
            with self.lock:
                to_average = self.average_dict.get(idx, cat_token)
                self.average_dict[idx] = np.mean([cat_token, to_average], axis=0)

    def fill_matrix(self):
        # Build matrix
        n_row = len(self.average_dict)
        n_cols = len(next(iter(self.average_dict.values())))
        self.matrix = np.zeros((n_row, n_cols))
        self.idx = np.zeros((n_row, 1), dtype=np.int)
        # Fill matrix
        for i, (idx, value) in enumerate(self.average_dict.items()):
            self.matrix[i] = value
            self.idx[i] = int(idx)
        # Standardize the Data
        self.matrix = StandardScaler().fit_transform(self.matrix)

    def transform(self):
        # Transform data
        pca = PCA(n_components=2)
        self.principal_components = pca.fit_transform(self.matrix)

    def build_df(self):
        data = np.concatenate([self.principal_components, self.idx], axis=1)
        # Create dataframe
        self.df = pd.DataFrame(
            data=data, columns=['principal component 1', 'principal component 2', 'idx'])

    def dump_df(self):
        filename = '{}{}.pca/PCA.pkl'.format(REPORTS_PATH, TARGET_DATASET)
        self.df.to_pickle(filename)

    @staticmethod
    def reshape_layers(encoded_layers):
        # Stack layers
        token_embeddings = torch.stack(encoded_layers, dim=0)
        # Remove batches
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        # Permute layers and tokens
        token_embeddings = token_embeddings.permute(1, 0, 2)
        # Return reshaped layers
        return token_embeddings

    @staticmethod
    def pooling_strategy(token):
        return torch.cat((token[-4],token[-3],token[-2],token[-1]))


def main():
    # Read all tweets
    print("> Reading dataset . . .")
    tweets = Dataset(TARGET_DATASET).extract()
    print("\tReading completed.")
    # PCA matrix
    print("> Computing matrix . . .")
    pca = Pca(tweets)
    pca.compute_matrix()
    print("\tComputing completed.")
    # PCA fit
    print("> Data transformation . . .")
    pca.transform()
    print("\tTransformation completed.")


main()