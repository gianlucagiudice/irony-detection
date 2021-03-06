import os
import re

from sklearn.utils import shuffle

from src.utils.config import REPORTS_PATH
from src.dataset.DataFrame import *
from src.utils.parameters import TARGET_DATASET
from src.training.Logger import Logger
from src.training.TrainingManager import TrainingManager, CLASSIFIER_LIST
from src.training.TrainingReport import TrainingReport


def create_report_folder(optional=""):
    report_path = REPORTS_PATH + TARGET_DATASET + optional
    if not os.path.exists(report_path):
        Path(report_path).mkdir(parents=True)


def extract_features_from_name(filename):
    return filename.split('.')[0].split('-')[1:]


def read_matrix_filename():
    files_path = DATASET_PATH_OUT + TARGET_DATASET
    files = [file for file in os.listdir(files_path)
             if file.split('.')[-1] == 'csv' and not re.match('labeled_tweets', file)]
    return sorted(files)


def main():
    # Title
    print("\t\t\t{} TRAINING MODELS ON DATASET: {} {}\n".format("=" * 5, TARGET_DATASET, "=" * 5))
    # Create report folder
    create_report_folder()
    # Create Logger
    logger = Logger().getLogger()
    # Read all matrix features
    matrix_file_list = read_matrix_filename()
    # Iterate over each features
    for idx_file, target_file in enumerate(matrix_file_list, 1):
        logger.print("\t\t%%%%% Target file {} - N. {}/{} %%%%%\n"
                     .format(target_file, idx_file, len(matrix_file_list)))
        # Read dataset for target features
        logger.print('Reading dataset . . .')
        path = '{}{}/{}'.format(DATASET_PATH_OUT, TARGET_DATASET, target_file)
        X, y = shuffle(*read_dataframe(path), random_state=22)
        logger.print('Read completed.')
        # Iterate over each classifier
        for idx_classifier, target_classifier in enumerate(CLASSIFIER_LIST, 1):
            # List of feature relative to file
            feature_list = extract_features_from_name(target_file)
            # Print status
            logger.print('>> Create model using features "{}": {}/{} - {} {}/{}'
                         .format('-'.join(feature_list), idx_file, len(matrix_file_list),
                                 type(target_classifier).__name__, idx_classifier, len(CLASSIFIER_LIST)))
            # Create training manager
            logger.print('> Start training on folds . . .')
            training_manager = TrainingManager(X, y, target_classifier)
            # Start training
            folds_report = training_manager.train_classifier()
            # Create report relative to classifier
            if folds_report:
                report = TrainingReport(feature_list, target_classifier, folds_report)
                report.create_report()
            # Print status
            logger.print('Training completed on each folds.\n\n')
    logger.print('>>> Models training completed using all features.')
    logger.completed()


if __name__ == "__main__":
    main()
