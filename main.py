from Tokenizer import Tokenizer

# Set tweets path
tweetsPath = "examples/example_tweets.txt"


def main():
    # Create tokenizer object
    toknz = Tokenizer(tweetsPath)
    # Compute all tweets
    toknz.computeTweets()
    # Print results
    toknz.printComparison()
    # Get list of words
    wordsList = toknz.getAllWords()


if __name__ == '__main__':
    main()
