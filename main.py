from language.Tokenizer import Tokenizer
from language.TextFeature import TextFeature
from language.PosFeature import PosFeature
from language.PragmaticParticlesFeature import PragmaticParticlesFeature

# Set tweets path
tweetsPath = "tests/example_tweets_small.txt"
# Parameters
debug = True


def main():
    # List of matrices
    matrices_list = []

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

    # ------ Compute text feature ------
    print("Text feature . . .")
    # Crete text feature object
    text_features = TextFeature(tokenizer.words)
    # Extract matrix from words list
    text_features.extractTermsMatrix()
    # Get terms matrix
    matrices_list.append(text_features.matrix)
    # Print results
    print(text_features if debug else "", end='')

    # ------ Pragmatic particles ------
    print("Pragmatic particles feature . . .")
    # Crete pragmatic particles object
    pragmatic_particles = PragmaticParticlesFeature(tweets)
    # Evaluate pragmatic particles for tweets
    pragmatic_particles.evaluatePragmaticParticles()
    # Get pragmatic particles matrix
    matrices_list.append(pragmatic_particles.matrix)
    # Print results
    print(pragmatic_particles if debug else "", end='')

    # ------ Part of speech features ------
    print("Pos tagging . . .")
    # Tag tweets
    part_of_speech = PosFeature(tweets)
    # Tag part of speech
    part_of_speech.computePosTags()
    # Get part of speech matrix
    matrices_list.append(part_of_speech.matrix)
    # Print results
    print(part_of_speech if debug else "", end='')


def readFile(path):
    with open(path) as file:
        return [tweet.strip() for tweet in file.readlines()]


if __name__ == '__main__':
    main()
