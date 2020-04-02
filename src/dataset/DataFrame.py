import csv

from src.Config import DATASET_PATH


class DataFrame:
    def __init__(self, dataset, text_feature, matrix):
        self.dataset = dataset
        self.textFeature = text_feature
        self.matrix = matrix

    def exportDataFrame(self):
        # Export labeled tweets
        self.exportLabeledTweets(self.dataset)
        # Export labeled matrix
        self.exportLabeledMatrix(self.matrix, self.dataset)

    def exportLabeledTweets(self, dataset):
        # Export path
        path = '{}{}/'.format(DATASET_PATH, dataset.datasetName)
        # Print progress
        print('\tSaving labeled tweets . . .')
        # Read all tweets in file
        with open('{}{}'.format(path, 'labeled_tweets.csv'), 'w') as csvfile:
            writer = csv.writer(csvfile)
            # Add header
            # writer.writerow(["tweet", "label"])
            # Write data
            for i, (tweet, label) in enumerate(zip(dataset.tweets, dataset.labels)):
                if i % 50 == 0:
                    print('\r\t\t{}% saved'.format(round(i / len(dataset.labels) * 100), 0), end='')
                writer.writerow([tweet] + [label])

    def exportLabeledMatrix(self, matrix, dataset):
        # Export path
        path = '{}{}/'.format(DATASET_PATH, dataset.datasetName)
        # Print progress
        print('\n\tSaving labeled matrix . . .')
        # Get text feature
        text_feature_file, unique_words = self.textFeature
        # Read all tweets in file
        with open('{}{}'.format(path, 'labeled_matrix.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Add header
            header = [word for word in unique_words] + \
                     ['feature_{}'.format(i+1) for i, _ in enumerate(matrix[0])] + \
                     ['label']
            # writer.writerow(header)
            # Write data
            with open(text_feature_file.name) as text_feature:
                for i, (matrix_row, label) in enumerate(zip(matrix, dataset.labels)):
                    if i % 50 == 0:
                        print('\r\t\t{}% saved'.format(round(i / len(dataset.labels) * 100), 0), end='')
                    text_row = [int(x) for x in text_feature.readline().strip().split(',')]
                    writer.writerow(text_row + list(matrix_row) + [label])
        text_feature_file.close()
