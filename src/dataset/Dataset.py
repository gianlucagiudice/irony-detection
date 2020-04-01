import json

from src.Config import DATASET_PATH


class Dataset:
    def __init__(self, dataset_name):
        self.datasetName = dataset_name
        self.ironicDict = None
        self.tweets = []
        self.labels = []

    def extract(self):
        # Extract dict containing ironic tweets files
        self.extractDict(self.datasetName)
        # Read tweets
        self.readTweets(self.datasetName, self.ironicDict)
        # Return data
        return self.tweets, self.labels

    def extractDict(self, dataset_name):
        path = '{}{}/files.json'.format(DATASET_PATH, dataset_name)
        with open(path) as json_dict:
            self.ironicDict = json.load(json_dict)

    def readTweets(self, dataset_name, file_dict):
        for file_name, label in file_dict.items():
            # Read all tweets in file
            path = '{}{}/files/{}'.format(DATASET_PATH, dataset_name, file_name)
            with open(path) as file:
                content = [tweet.strip() for tweet in file.readlines()]
                self.tweets += content
                self.labels += [label] * len(content)
