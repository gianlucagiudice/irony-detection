from Tokenizer import Tokenizer

def main():
    # Set tweets path
    tweetsPath = "examples/example_tweets.txt"

    # Create tokenizer object
    toknz = Tokenizer(tweetsPath)

    # Compute all tweets
    toknz.computeTweets()
    # Print results
    toknz.printComparison()


if __name__ == '__main__':
    main()
