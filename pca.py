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
from src.utils.config import REPORTS_PATH, THREAD_NUMBER, DATASET_PATH_OUT
from src.utils.parameters import TARGET_DATASET

COMPUTE_MATRIX = False


class Pca:

    def __init__(self, tweets=None):
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
        # Lock
        self.lock = threading.Lock()

    def compute_matrix(self):
        # Compute dict
        self.compute_dict()
        # Create matrix
        self.fill_matrix()
        # Dump matrix
        self.dump_df(
            self.build_df(self.matrix, self.idx), '{}{}.pca/PCA_matrix.pkl'.format(DATASET_PATH_OUT, TARGET_DATASET))

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

    def transform(self, n_components):
        # Transform data
        pca = PCA(n_components=n_components)
        self.principal_components = pca.fit_transform(self.matrix)
        # Create dataframe
        columns = ['principal component {}'.format(i) for i in range(1, n_components + 1)] + ['idx']
        df = self.build_df(self.principal_components, self.idx, columns=columns)
        # Save dataframe
        self.dump_df(df, '{}{}.pca/PCA_{}D.pkl'.format(REPORTS_PATH, TARGET_DATASET, n_components))

    def load_matrix(self):
        df = pd.read_pickle("{}{}.pca/PCA_matrix.pkl".format(DATASET_PATH_OUT, TARGET_DATASET))
        self.matrix = df.iloc[:,:-1].values
        self.idx = np.array([[x] for x in df.iloc[:,-1].values])

    @staticmethod
    def build_df(data, idx, columns=None):
        data = np.concatenate([data, idx], axis=1)
        # Create dataframe
        return pd.DataFrame(data=data, columns=columns)

    @staticmethod
    def dump_df(df, path):
        df.to_pickle(path)


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
    if COMPUTE_MATRIX:
        pca = Pca(tweets)
        pca.compute_matrix()
    else:
        pca = Pca()
        pca.load_matrix()
    print("\tComputing completed.")
    # PCA fit
    print("> Data transformation 2D. . .")
    pca.transform(n_components=2)
    print("\tTransformation completed.")
    # PCA fit
    print("> Data transformation 3D. . .")
    pca.transform(n_components=3)
    print("\tTransformation completed.")


main()