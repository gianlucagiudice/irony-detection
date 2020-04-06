import csv

from src.Config import DATASET_PATH


class DataFrame:
    def __init__(self, dataset, text_feature, matrix, target_feature):
        self.dataset = dataset
        self.text_feature = text_feature
        self.matrix = matrix
        self.target_feature = target_feature

    def export_data_frame(self):
        # Export labeled tweets
        self.export_labeled_tweets(self.dataset)
        # Export labeled matrix
        self.export_labeled_matrix(self.matrix, self.dataset)

    def export_labeled_tweets(self, dataset):
        # Export path
        path = '{}{}/'.format(DATASET_PATH, dataset.dataset_name)
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

    def export_labeled_matrix(self, matrix, dataset):
        # Export path
        path = '{}{}/'.format(DATASET_PATH, dataset.dataset_name)
        # Print progress
        print('\n\tSaving labeled matrix . . .')
        # Get text feature
        text_feature_file, unique_words = self.text_feature
        # Read all tweets in file
        file_name = 'labeled_matrix-{}.csv'.format('-'.join(self.target_feature.keys()))
        with open('{}{}'.format(path, file_name), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Add header
            header = ['t_{}'.format(word) for word in unique_words] + \
                     ['feature_{}'.format(i+1) for i, _ in enumerate(matrix[0])] + \
                     ['label']
            writer.writerow(header)
            # Write data
            with open(text_feature_file.name) as text_feature:
                for i, (matrix_row, label) in enumerate(zip(matrix, dataset.labels)):
                    if i % 50 == 0:
                        print('\r\t\t{}% saved'.format(round(i / len(dataset.labels) * 100), 0), end='')
                    text_row = [int(x) for x in text_feature.readline().strip().split(',')]
                    writer.writerow(text_row + list(matrix_row) + [label])
        text_feature_file.close()
