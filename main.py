import numpy as np
from language.features.Tokenizer import Tokenizer
from language.features.TextFeature import TextFeature
from language.features.PosFeature import PosFeature
from language.features.PragmaticParticlesFeature import PragmaticParticlesFeature
from language.features.EmotionalFeature import EmotionalFeature


# Set tweets path
tweetsPath = "tests/example_tweets_small.txt"
# Parameters
debug = True


def main():
    # List of matrices
    matrices = []

    # ------ Read file ------
    print("Reading file . . .")
    tweets = readFile(tweetsPath)

    # ------ Tokenizer ------
    print("Tokenizer . . .")
    # Create tokenizer object
    tokenizer = Tokenizer(tweets)
    # Compute all tweets
    tokenizer.evaluateTweets()
    # Print results
    print(tokenizer if debug else "", end='')

    # ------ Compute text features ------
    print("Text features . . .")
    # Crete text features object
    text_features = TextFeature(tokenizer.words)
    # Get terms matrix
    matrices.append(text_features.extractTermsMatrix())
    # Print results
    print(text_features if debug else "", end='')

    # ------ Pragmatic particles ------
    print("Pragmatic particles features . . .")
    # Crete pragmatic particles object
    pragmatic_particles = PragmaticParticlesFeature(tweets)
    # Evaluate pragmatic particles for tweets
    matrices.append(pragmatic_particles.evaluatePragmaticParticles())
    # Print results
    print(pragmatic_particles if debug else "", end='')

    # ------ Part of speech features ------
    print("Pos tagging . . .")
    # Create pos tagger object
    part_of_speech = PosFeature(tweets)
    # Tag part of speech
    matrices.append(part_of_speech.computePosTags())
    # Print results
    print(part_of_speech if debug else "", end='')

    #  ------ Emotional features ------
    print("Emotional feature . . .")
    # Create emotional feature object
    emotional = EmotionalFeature(tokenizer.words)
    # Evaluate emotional feature
    matrices.append(emotional.evaluateEmotions())
    # Print results
    print(emotional if debug else "", end='')

    #  ------ Build complete matrix ------
    print("Builind matrix . . .")
    # Build matrix
    matrix = buildMatrix(matrices)
    # Print results
    print(matrix if debug else "", end='')


def readFile(path):
    with open(path) as file:
        return [tweet.strip() for tweet in file.readlines()]


def buildMatrix(matrices):
    return np.concatenate(matrices, axis=1)


if __name__ == '__main__':
    main()
