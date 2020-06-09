import numpy as np
import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.dataset.Dataset import Dataset
from src.features.text.Bert import Bert
from src.utils.config import REPORTS_PATH, DATASET_PATH_OUT
from src.utils.parameters import TARGET_DATASET

COMPUTE_MATRIX = True

valid_token_regexp = "[a-zA-Z]+"


def dataset_type_name(dataset_type):
    return '+'.join([key for key, value in dataset_type.items() if value])


class Pca:

    def __init__(self, tweets=None, labels=None):
        # Tweets
        self.tweet_list = tweets
        self.labels = labels
        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Model
        self.model = Bert.build_model()
        # Matrix dict
        self.word_embedding_dict = dict()
        # Matrix
        self.matrix = None
        # Idx
        self.idx = None
        self.words_vector = None
        # Coefficient
        self.coefficient_dict = dict()
        self.coefficient_matrix = None

    def compute_matrix(self, target_dataset):
        # Compute dict
        self.compute_word_embedding()
        # Create matrix
        self.fill_matrix()
        # Create coefficient matrix
        self.evaluate_coefficient_vector()
        # Dump matrix
        filename = 'out_matrix${}'.format(dataset_type_name(target_dataset))
        path = '{}{}.pca/{}.pkl'.format(DATASET_PATH_OUT, TARGET_DATASET, filename)
        columns = ["embedding_{}".format(i) for i in range(1, np.size(self.matrix, 1) + 1)] +\
                  ['#+', '#-', '#', 'coeff']
        self.dump_df(self.build_df([self.matrix, self.coefficient_matrix],
                                   columns=columns, index=self.words_vector.flatten()), path)

    def compute_word_embedding(self):
        for index, tweet in enumerate(self.tweet_list):
            self.compute_row(tweet, index)
            if index % 100 == 0:
                print("\r\t\t{}% completed".format((index+1) / len(self.tweet_list) * 100), end='')
        print("\r\t\t{}% completed".format(100))

    def fill_matrix(self):
        # Build matrix
        n_row = len(self.word_embedding_dict)
        n_cols = len(next(iter(self.word_embedding_dict.values())))
        self.matrix = np.zeros((n_row, n_cols))
        self.idx = np.zeros((n_row, 1), dtype=np.int)
        # Fill matrix
        for i, (idx, value) in enumerate(self.word_embedding_dict.items()):
            self.matrix[i] = value
            self.idx[i] = int(idx)
        # Standardize the Data
        self.matrix = StandardScaler().fit_transform(self.matrix)

    def evaluate_coefficient_vector(self):
        # Initialize coefficient vector
        self.coefficient_matrix = np.zeros((len(self.word_embedding_dict), 4))
        words_vector = []
        # Evaluate vector
        for i, word_idx in enumerate(self.idx):
            pos, neg = self.coefficient_dict[word_idx[0]]['+'], self.coefficient_dict[word_idx[0]]['-']
            self.coefficient_matrix[i] = [pos, neg, pos + neg, (pos/(pos + neg)) - (neg/(pos + neg))]
            words_vector.append(self.tokenizer.convert_ids_to_tokens(word_idx)[0])
        self.words_vector = np.array([[word] for word in words_vector])

    def compute_row(self, tweet, row_idx):
        # Tokenize tweet
        indexed_tokens, segments_ids = Bert.tokenize(
            tweet, self.tokenizer, exclude_hot_words=False, token_regexp=valid_token_regexp)
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
            # Update word embeddings
            to_average = self.word_embedding_dict.get(idx, cat_token)
            self.word_embedding_dict[idx] = np.mean([cat_token, to_average], axis=0)
            # Update coefficient dict
            dict_to_update = self.coefficient_dict.get(idx, {'+': 0, '-': 0})
            dict_to_update['+'] = dict_to_update['+'] + (self.labels[row_idx] == 'ironic')
            dict_to_update['-'] = dict_to_update['-'] + (self.labels[row_idx] == 'non_ironic')
            self.coefficient_dict.update([(idx, dict_to_update)])

    def transform(self, n_components):
        # Transform data
        pca = PCA(n_components=n_components)
        # x = pca.fit(self.matrix)
        principal_components = pca.fit_transform(self.matrix)
        # Create dataframe
        columns = ['principal component {}'.format(i) for i in range(1, n_components + 1)] +\
                  ['idx', 'word', '#+', '#-', '#', 'coefficient']
        data = [principal_components, self.idx, self.words_vector, self.coefficient_matrix]
        types = {'idx': int, 'word': np.str, '#+': float, '#-': float, '#': float, 'coefficient': float}
        types.update([('principal component {}'.format(i), float) for i in range(1, n_components + 1)])
        df = self.build_df(data, types=types, columns=columns)
        # Save dataframe
        self.dump_df(df, '{}{}.pca/PCA_{}D.pkl'.format(REPORTS_PATH, TARGET_DATASET, n_components))

    def transform_transposed(self, n_components):
        # Transform data
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(self.matrix.transpose())
        # Create dataframe
        columns = ['principal component {}'.format(i) for i in range(1, n_components + 1)]
        types = {'principal component {}'.format(i): float for i in range(1, n_components + 1)}
        df = self.build_df([principal_components], types=types, columns=columns)
        # Save dataframe
        self.dump_df(df, '{}{}.pca/PCA_{}D_transposed.pkl'.format(REPORTS_PATH, TARGET_DATASET, n_components))

    def load_matrix(self):
        df = pd.read_pickle("{}{}.pca/out_matrix.pkl".format(DATASET_PATH_OUT, TARGET_DATASET))
        self.matrix = df.iloc[:,:-1].values
        self.idx = np.array([[x] for x in df.iloc[:,-1].values])

    @staticmethod
    def build_df(data_list, types=None, columns=None, index=None):
        data = np.concatenate([*data_list], axis=1)
        # Create dataframe
        if types:
            return pd.DataFrame(data=data, columns=columns, index=index).astype(types)
        else:
            return pd.DataFrame(data=data, columns=columns, index=index)

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
        return torch.cat((token[-4], token[-3], token[-2], token[-1]))


def compute_target_dataset(dataset, target_dataset):
    tweets = dataset.extract(target_dataset=target_dataset)
    labels = dataset.labels
    # PCA matrix
    print("\t> Computing matrix . . .")
    if COMPUTE_MATRIX:
        pca = Pca(tweets=tweets, labels=labels)
        pca.compute_matrix(target_dataset=target_dataset)
    else:
        pca = Pca()
        pca.load_matrix()
    print("\t\tComputing completed.")
    # PCA fit 2D
    print("\t> Data transformation 2D. . .")
    pca.transform(n_components=2)
    print("\t\tTransformation completed.")
    # PCA fit 3D
    print("\t> Data transformation 3D. . .")
    pca.transform(n_components=3)
    print("\t\tTransformation completed.")
    # PCA fit 2D transposed
    print("\t> Data transformation 2D (transposed). . .")
    pca.transform_transposed(n_components=2)
    print("\t\tTransformation completed.")


def main():
    # Read all tweets
    print("> Reading dataset . . .")
    print("\tReading completed.")
    dataset = Dataset(TARGET_DATASET)
    # Combinations of dataset
    # IMPORTANT! Do not reorder items
    target_dataset_list = [{'ironic': True, 'non_ironic': False},
                           {'ironic': False, 'non_ironic': True},
                           {'ironic': True, 'non_ironic': True}]
    # Compute each dataset separately
    for i, target_dataset in enumerate(target_dataset_list, 1):
        print("% Target dataset: {} - ({}/{})".format(target_dataset, i, len(target_dataset_list)))
        compute_target_dataset(dataset, target_dataset=target_dataset)


main()
