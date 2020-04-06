import sys

from src.features.FeatureManager import FeatureManager
from src.dataset.Dataset import Dataset
from src.dataset.DataFrame import DataFrame


# Parameters
DEBUG = False
# Dataset
TARGET_DATASET = 'TwReyes2013'


def main():
    # ------ Read dataset ------
    print(">> Reading dataset . . .")
    dataset = Dataset(TARGET_DATASET)
    tweets, labels = dataset.extract()

    # ------ Build features matrix ------
    target_features = extract_target_features()
    print(">> Extracting features . . .")
    feature_extractor = FeatureManager(tweets, target_features, debug=DEBUG)
    text_feature, matrix = feature_extractor.extract_features()

    # ------ Export data frame to file ------
    print(">> Saving features . . .")
    df = DataFrame(dataset, text_feature, matrix, target_features)
    df.export_data_frame()

    print('Completed.')


def extract_target_features():
    target_feature = {'pp': False, 'pos': False, 'emot': False}
    parameters = set(sys.argv[1:])
    if len(sys.argv) > 1:
        if not parameters.issubset(target_feature.keys()):
            print('ERROR! Invalid features passed as parameters')
            quit(-1)
        for parameter in parameters:
            target_feature[parameter] = True
    return target_feature


if __name__ == '__main__':
    main()
