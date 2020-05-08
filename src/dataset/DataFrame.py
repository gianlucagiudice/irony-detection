import csv
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATASET_PATH_OUT


def read_dataframe(filename):
    n_cols = pd.read_csv(filename, nrows=1).values.size - 1
    n_rows = sum(1 for _ in open(filename)) - 1
    # Create matrix
    X = np.zeros((n_rows, n_cols), dtype=np.float)
    y = np.zeros(n_rows, dtype=np.bool)
    # Read csv
    chuncksize = 2500
    print('\t{}% completed . . .'.format(0), end='')
    for idx, chunk in enumerate(pd.read_csv(filename, chunksize=chuncksize)):
        for idx_chunk, (X_row, y_row) in enumerate(zip(chunk.values[:, :-1], chunk.values[:, -1])):
            X[idx * chuncksize + idx_chunk] = np.array(X_row, dtype=np.float)
            label = True if y_row == 'ironic' else False
            y[idx * chuncksize + idx_chunk] = np.array(label, dtype=np.bool)
        print('\r\t{}% completed . . .'. format(round((idx+1) * chuncksize / n_rows * 100), 3), end='')
    print('\r\t{}% completed.'.format(100))
    return X, y


class DataFrame:
    def __init__(self, dataset, matrix_dict):
        self.dataset = dataset
        # Unpack text feature
        self.file, self.header, self.text_feature_name = matrix_dict['text']
        self.matrix_dict = matrix_dict

    def save_data_frame(self):
        # Create dataset folder
        self.create_folder()
        # Export labeled tweets
        self.export_labeled_tweets()
        # Export labeled matrix
        self.export_labeled_matrix()

    def create_folder(self):
        # Dataset folder
        path = '{}{}/'.format(DATASET_PATH_OUT, self.dataset.dataset_name)
        Path(path).mkdir(parents=True, exist_ok=True)

    def export_labeled_matrix(self):
        # Save all dataframes
        print('\n\t> Saving labeled dataframe . . .', end='')
        powerset_features = self.powerset(list(self.matrix_dict.keys()))
        for idx, set_features in enumerate(powerset_features, 1):
            print('\n', end='')
            target_feature = self.extract_target_features(set_features)
            matrix = self.build_matrix(target_feature)
            print('\t\t- Saving dataframe N. {}/{}: {}'.format(idx, len(powerset_features), target_feature))
            self.save_matrix(self.file, matrix, target_feature)
        # Close file
        self.file.close()
        print(end='\n')

    def save_matrix(self, text_feature_file, matrix, target_feature):
        # Export path
        path = '{}{}/'.format(DATASET_PATH_OUT, self.dataset.dataset_name)
        # Generate filename
        file_name = self.create_filename(target_feature)
        # Create file
        with open('{}{}'.format(path, file_name), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Add header
            header = ['t_{}'.format(word) for word in self.header] if target_feature['text'] else []
            header = header + ['f_{}'.format(i + 1) for i, _ in enumerate(matrix[0])]
            header = header + ['label']
            writer.writerow(header)
            # Write data
            if target_feature['text']:
                with open(text_feature_file.name) as text_feature:
                    self.write_matrix_rows(writer, matrix, text_feature=text_feature)
            else:
                self.write_matrix_rows(writer, matrix)
        print('\r\t\t\t{}% saved'.format(100), end='')

    def write_matrix_rows(self, writer, matrix, text_feature=None):
        for i, (matrix_row, label) in enumerate(zip(matrix, self.dataset.labels)):
            if i % 50 == 0:
                print('\r\t\t\t{}% saved'.format(round(i / len(self.dataset.labels) * 100), 0), end='')
            text_row = [float(x) for x in text_feature.readline().strip().split(',')] if text_feature else []
            writer.writerow(text_row + list(matrix_row) + [label])

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
        temp_dict = target_feature.copy()
        temp_dict.pop('text')
        for feature, use in temp_dict.items():
            if use:
                matrix = np.concatenate([matrix, self.matrix_dict[feature]], axis=1)
        return matrix

    @staticmethod
    def powerset(iterable):
        out = []
        x = len(iterable)
        for i in range(1 << x):
            out += [[iterable[j] for j in range(x) if (i & (1 << j))]]
        # Remove empty set
        out.remove([])
        # Return powerset
        return out

    def extract_target_features(self, feature_list):
        target_feature = {feature: False for feature in self.matrix_dict.keys()}
        for feature in feature_list:
            target_feature[feature] = True
        return target_feature

    def create_filename(self, target_feature):
        keys = [key for key, value in target_feature.items() if value]
        if 'text' in keys:
            keys[keys.index('text')] = self.text_feature_name
        return '-'.join(['labeled_matrix'] + keys) + '.csv'