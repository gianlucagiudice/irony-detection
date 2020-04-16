from src.config import REPORTS_PATH
from src.training.TrainingManager import TrainingManager, CLASSIFIER_LIST
from src.training.TrainingReport import TrainingReport
from src.training.Logger import Logger
from src.parameters import TARGET_DATASET, TARGET_TEXT_FEATURE

from sklearn.utils import shuffle
import os
import shutil

from src.dataset.DataFrame import *


def create_report_folder():
    report_path = REPORTS_PATH + TARGET_DATASET
    if os.path.exists(report_path):
        shutil.rmtree(report_path)
    Path(report_path).mkdir(parents=True)


def extract_features_from_name(filename):
    return filename.split('.')[0].split('-')[1:]


def read_matrix_filename():
    return [file for file in os.listdir(DATASET_PATH_OUT + TARGET_DATASET)
            if 'labeled_matrix-{}'.format(TARGET_TEXT_FEATURE) in file]


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
        # Iterate over each classifier
        for idx_classifier, target_classifier in enumerate(CLASSIFIER_LIST, 1):
            # List of feature relative to file
            feature_list = extract_features_from_name(target_file)
            # Print status
            logger.print('>> Create model using features "{}": {}/{} - {} {}/{}'
                         .format('-'.join(feature_list), idx_file, len(matrix_file_list),
                                 type(target_classifier).__name__, idx_classifier, len(CLASSIFIER_LIST)))
            # Read dataset for target features
            logger.print('Reading dataset . . .')
            path = '{}{}/{}'.format(DATASET_PATH_OUT, TARGET_DATASET, target_file)
            X, y = shuffle(*read_data_frame(path), random_state=22)
            logger.print('Read completed.')
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


main()
