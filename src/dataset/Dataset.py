import json

from src.Config import DATASET_PATH_IN
from src.Config import DATASET_LABEL_NAME


class Dataset:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.ironic_dict = None
        self.tweets = []
        self.labels = []

    def extract(self):
        # Extract dict containing ironic tweets files
        self.extract_dict(self.dataset_name)
        # Read tweets
        self.read_tweets(self.dataset_name, self.ironic_dict)
        # Return data
        return self.tweets, self.labels

    def extract_dict(self, dataset_name):
        path = '{}{}/{}.json'.format(DATASET_PATH_IN, dataset_name, DATASET_LABEL_NAME)
        with open(path) as json_dict:
            self.ironic_dict = json.load(json_dict)

    def read_tweets(self, dataset_name, file_dict):
        for file_name, label in file_dict.items():
            # Read all tweets in file
            path = '{}{}/{}'.format(DATASET_PATH_IN, dataset_name, file_name)
            with open(path) as file:
                content = [tweet.strip() for tweet in file.readlines()]
                self.tweets += content
                self.labels += [label] * len(content)
