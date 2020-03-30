from language.features.FeatureManager import FeatureManager


# Set tweets path
tweetsPath = "tests/example_tweets_small.txt"
# Parameters
DEBUG = True


def main():
    # ------ Read file ------
    print("Reading file . . .")
    tweets = readFile(tweetsPath)

    # ------ Build features matrix ------
    print("Extracting features . . .")
    feature_extractor = FeatureManager(tweets, debug=DEBUG)
    matrix = feature_extractor.extractFeatures()
    print(matrix if DEBUG else "", end='')


def readFile(path):
    with open(path) as file:
        return [tweet.strip() for tweet in file.readlines()]


if __name__ == '__main__':
    main()
