from Tokenizer import Tokenizer

# Set tweets path
tweetsPath = "examples/example_tweets.txt"

# Create tokenizer object
toknz = Tokenizer(tweetsPath)

# Compute all tweets
toknz.computeTweets()
# Print results
toknz.printComparison()