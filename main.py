from src.dataset.DataFrame import DataFrame
from src.dataset.Dataset import Dataset
from src.features.FeatureManager import FeatureManager
from src.utils.parameters import TARGET_TEXT_FEATURE, TARGET_DATASET, TEXT_FEATURE_STRATEGY


def main():
    # ----- Title -----
    print("\t\t\t{} COMPUTE FEATURES ON DATASET: {} - ({}) {}"
          .format('='*5, TARGET_DATASET, TARGET_TEXT_FEATURE, '='*5), end='\n\n')
    # ----- Read dataset -----
    print(">> Reading dataset . . .")
    dataset = Dataset(TARGET_DATASET)
    dataset.extract()
    # ----- Compute matrix -----
    print(">> Extracting features . . .")
    feature_extractor = FeatureManager(dataset.tweets, TEXT_FEATURE_STRATEGY, debug=False)
    feature_matrix = feature_extractor.extract_features()
    # ----- Export data frame to file -----
    print(">> Saving features . . .")
    df = DataFrame(dataset, feature_matrix)
    df.save_data_frame()
    # ----- Process completed -----
    print('>> Completed.\n')


if __name__ == '__main__':
    main()
