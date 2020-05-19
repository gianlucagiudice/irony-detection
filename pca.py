import csv

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer

from src.dataset.DataFrame import create_folder
from src.dataset.Dataset import Dataset
from src.features.text.Bert import Bert
from src.utils.config import DATASET_PATH_OUT
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

    def compute_matrix(self):
        # Compute dict
        self.compute_dict()
        # Write dict to file
        self.dump_dict()

    def compute_dict(self):
        for tweet in self.tweet_list:
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
                to_average = self.average_dict.get(idx, cat_token)
                self.average_dict[idx] = np.mean([cat_token, to_average], axis=0)

    def dump_dict(self):
        path = '{}{}.pca/'.format(DATASET_PATH_OUT, TARGET_DATASET)
        # Create folder for target path
        create_folder('{}.pca'.format(TARGET_DATASET))
        file_name = '{}{}.csv'.format(path, 'PCA')
        # Write to file
        with open(file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            [writer.writerow(list(value) + [idx]) for idx, value in self.average_dict.items()]


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
    print("Reading dataset . . .")
    tweets = Dataset(TARGET_DATASET).extract()
    print("Reading completed.")
    # PCA
    print("Computing matrix . . .")
    pca = Pca(tweets)
    pca.compute_matrix()
    print("Computing completed")

    x = 0

main()