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
    print(toknz if debug else "")
    # Get list of words
    wordsList = toknz.getAllWords()

    # Crete text feature object
    txtf = TextFeature(wordsList)
    # Extract matrix from words list
    txtf.extractMatrix()



if __name__ == '__main__':
    main()
