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

    #tweets = tweets[:5]

    # ------ Build features matrix ------
    print(">> Extracting features . . .")
    feature_extractor = FeatureManager(tweets, debug=DEBUG)
    text_feature, matrix = feature_extractor.extractFeatures()

    # ------ Export data frame to file ------
    print(">> Saving features . . .")
    df = DataFrame(dataset, text_feature, matrix)
    df.exportDataFrame()


if __name__ == '__main__':
    main()
