import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import gc

from src.config import DATASET_PATH_OUT
from src.features.FeatureManager import FEATURES


def create_filename(name, target_feature, ext):
    keys = [x for x in target_feature.keys() if target_feature[x] is True]
    sep_char = '-' if len(keys) > 0 else ''
    file_name = '{}-bow{}{}.{}'.format(name, sep_char, '-'.join(keys), ext)
    return file_name


def read_data_frame(filename):
    df = pd.read_csv(filename)
    X = np.array(df.values[:, :-1], dtype=np.uint8)
    y = np.array([True if value == 'ironic' else False for value in df.values[:, -1]], dtype=np.bool)
    del df
    gc.collect()
    return X, y


class DataFrame:
    def __init__(self, dataset, text_feature, matrix_dict):
        self.dataset = dataset
        self.text_feature = text_feature
        self.matrix_dict = matrix_dict

    def save_data_frame(self):
        # Create dataset folder
        self.create_folder()
        # Export labeled tweets
        self.export_labeled_tweets()
        # Export labeled matrix
        self.export_labeled_matrix()

    def create_folder(self):
        path = '{}{}/'.format(DATASET_PATH_OUT, self.dataset.dataset_name)
        Path(path).mkdir(parents=True, exist_ok=True)

    def export_labeled_matrix(self):
        # Save all dataframes
        print('\n\t> Saving labeled dataframe . . .', end='')
        text_feature_file, _ = self.text_feature
        powerset_features = self.powerset(FEATURES)
        for idx, set_features in enumerate(powerset_features, 1):
            print('\n', end='')
            target_feature = self.extract_target_features(set_features)
            matrix = self.build_matrix(target_feature)
            print('\t\t- Saving dataframe N. {}/{}: {}'.format(idx, len(powerset_features), target_feature))
            self.save_matrix(text_feature_file, matrix, target_feature)
        # Close file
        text_feature_file.close()
        print(end='\n')

    def save_matrix(self, text_feature_file, matrix, target_feature):
        # Export path
        path = '{}{}/'.format(DATASET_PATH_OUT, self.dataset.dataset_name)
        # Generate filename
        file_name = create_filename('labeled_matrix', target_feature, 'csv')
        # Create file
        with open('{}{}'.format(path, file_name), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Add header
            _, unique_words = self.text_feature
            header = ['t_{}'.format(word) for word in unique_words] + \
                     ['f_{}'.format(i + 1) for i, _ in enumerate(matrix[0])] + \
                     ['label']
            writer.writerow(header)
            # Write data
            with open(text_feature_file.name) as text_feature:
                for i, (matrix_row, label) in enumerate(zip(matrix, self.dataset.labels)):
                    if i % 50 == 0:
                        print('\r\t\t\t{}% saved'.format(round(i / len(self.dataset.labels) * 100), 0), end='')
                    text_row = [int(x) for x in text_feature.readline().strip().split(',')]
                    writer.writerow(text_row + list(matrix_row) + [label])
        print('\r\t\t\t{}% saved'.format(100), end='')

    def export_labeled_tweets(self):
        print('\t> Saving labeled tweets . . .')
        # Export path
        path = '{}{}/'.format(DATASET_PATH_OUT, self.dataset.dataset_name)
        # Read all tweets in file
        with open('{}{}'.format(path, 'labeled_tweets.csv'), 'w') as csvfile:
            writer = csv.writer(csvfile)
            # Add header
            writer.writerow(["tweet", "label"])
            # Write data
            for i, (tweet, label) in enumerate(zip(self.dataset.tweets, self.dataset.labels)):
                if i % 50 == 0:
                    print('\r\t\t{}% saved'.format(round(i / len(self.dataset.labels) * 100), 0), end='')
                writer.writerow([tweet] + [label])
        print('\r\t\t{}% saved'.format(100), end='')

    def build_matrix(self, target_feature):
        matrix = [[] for _ in range(len(self.dataset.tweets))]
        # Concatenate features
        for feature, to_use in target_feature.items():
            if to_use:
                matrix = np.concatenate([matrix, self.matrix_dict[feature]], axis=1)
        return matrix

    @staticmethod
    def powerset(iterable):
        out = []
        x = len(iterable)
        for i in range(1 << x):
            out += [[iterable[j] for j in range(x) if (i & (1 << j))]]
        return out

    @staticmethod
    def extract_target_features(feature_list):
        target_feature = {'pp': False, 'pos': False, 'emot': False}
        for feature in feature_list:
            target_feature[feature] = True
        return target_feature
