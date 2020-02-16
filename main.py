from language.Tokenizer import Tokenizer
from language.TextFeature import TextFeature
from language.PosFeature import PosFeature
from language.ParseInitialism import ParseInitialism

# Set tweets path
tweetsPath = "tests/example_tweets_small.txt"
# Parameters
debug = True


def main():
    # Read file
    print("Reading file . . .")
    tweets = readFile(tweetsPath)

    # Parse initialism
    print("Parsing initialism . . .")
    # Create parser object
    init = ParseInitialism(tweets)
    # Parse initialism
    init.parseTweets()
    # Print results
    print(init if debug else "", end='')
    # Get tweets parsed
    tweets_parsed = init.tweetsParsed

    # Tokenizer
    print("Tokenizer . . .")
    # Create tokenizer object
    toknz = Tokenizer(tweets_parsed)
    # Compute all tweets
    toknz.computeTweets()
    # Print results
    print(toknz if debug else "", end='')
    # Get list of words
    words_list = toknz.words

    # Compute text feature
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
    pos = PosFeature(toknz.words, tweets)
    # Tag part of speech
    pos.computePosTag()
    # Print results
    print(pos if debug else "", end='')


def readFile(path):
    with open(path) as file:
        return [tweet.strip() for tweet in file.readlines()]


if __name__ == '__main__':
    main()