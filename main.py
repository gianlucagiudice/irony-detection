import subprocess

from language.Tokenizer import Tokenizer
from language.TextFeature import TextFeature
from language.PosFeature import PosFeature

# Set tweets path
tweetsPath = "tests/example_tweets_small.txt"
# Parameters
debug = True


def main():
    # Tokenizer process
    print("Tokenizer . . .")
    # Create tokenizer object
    toknz = Tokenizer(tweetsPath)
    # Compute all tweets
    toknz.computeTweets()
    # Print results
    print(toknz if debug else "", end='')
    # Get list of words
    words_list = toknz.words

    # Evaluate text feature
    print("Text feature . . .")
    # Crete text feature object
    txtf = TextFeature(words_list)
    # Extract matrix from words list
    txtf.extractMatrix()
    # Print results
    print(txtf if debug else "", end='')

    # Words tagging
    print("Pos tagging . . .")
    # Tag tweets
    pos = PosFeature(toknz.words, toknz.tweets)
    # Tag part of speech
    pos.computePosTag()
    # Print results
    print(pos if debug else "", end='')


if __name__ == '__main__':
    main()
