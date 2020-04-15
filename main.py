from src.features.FeatureManager import FeatureManager
from src.dataset.Dataset import Dataset
from src.dataset.DataFrame import DataFrame
from src.parameters import TARGET_DATASET, TEXT_FEATURE_STRATEGY, DEBUG


def main():
    # ----- Read dataset -----
    print(">> Reading dataset . . .")
    dataset = Dataset(TARGET_DATASET)
    dataset.extract()
    # ----- Compute matrix -----
    print(">> Extracting features . . .")
    feature_extractor = FeatureManager(dataset.tweets, TEXT_FEATURE_STRATEGY, debug=DEBUG)
    text_feature, matrix = feature_extractor.extract_features()
    # ----- Export data frame to file -----
    print(">> Saving features . . .")
    df = DataFrame(dataset, text_feature, matrix)
    df.save_data_frame()
    # ----- Process completed -----
    print('>> Completed.')


if __name__ == '__main__':
    main()
