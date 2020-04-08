from src.features.FeatureManager import FeatureManager
from src.dataset.Dataset import Dataset
from src.dataset.DataFrame import DataFrame


# Parameters
DEBUG = False
# Dataset
TARGET_DATASET = 'TwReyes2013'
TARGET_DATASET = 'Test'
# List of features
FEATURES = ['pp', 'pos', 'emot']


def main():
    # ----- Read dataset -----
    print(">> Reading dataset . . .")
    dataset = Dataset(TARGET_DATASET)
    dataset.extract()
    # ----- Compute matrix -----
    print(">> Extracting features . . .")
    feature_extractor = FeatureManager(dataset.tweets, debug=DEBUG)
    text_feature, matrix = feature_extractor.extract_features()
    # ----- Export data frame to file -----
    print(">> Saving features . . .")
    df = DataFrame(dataset, text_feature, matrix)
    df.save_data_frame()
    # ----- Process completed -----
    print('>> Completed.')


def compute_matrix(dataset, target_features):
    # Compute matrix
    print(">> Extracting features . . .")
    feature_extractor = FeatureManager(dataset.tweets, target_features, debug=DEBUG)
    text_feature, matrix = feature_extractor.extract_features()
    # Export data frame to file
    print(">> Saving features . . .")
    df = DataFrame(dataset, text_feature, matrix, target_features)
    df.save_data_frame()
    # Process completed
    print('>> Completed.')


if __name__ == '__main__':
    main()
