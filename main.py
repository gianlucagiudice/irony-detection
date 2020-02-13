from Tokenizer import Tokenizer
from TextFeature import TextFeature

# Set tweets path
tweetsPath = "examples/example_tweets_small.txt"
# Parameters
debug = True


def main():
    # Create tokenizer object
    toknz = Tokenizer(tweetsPath)
    # Compute all tweets
    toknz.computeTweets()
    # Print results
    print(toknz if debug else "", end='')
    # Get list of words
    words_list = toknz.words

    # Crete text feature object
    txtf = TextFeature(words_list)
    # Extract matrix from words list
    txtf.extractMatrix()
    # Print results
    print(txtf if debug else "", end='')


if __name__ == '__main__':
    main()
