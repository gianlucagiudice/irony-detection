import re
from nltk.tokenize import TweetTokenizer


class Tokenizer:
    def __init__(self, tweets):
        # List of tweets
        self.tweets = tweets
        # List of tweets tokenized
        self.tokens = None
        # List of words in each tweet
        self.words = None

    def evaluateTweets(self):
        self.tokenizeTweets()
        self.extractWords()

    def tokenizeTweets(self):
        # Construct tokenizer object
        preserve_case = False  # Convert each work to lowercase
        reduce_len = True  # mooooooonkey -> mooonkey
        tokenizer = TweetTokenizer(preserve_case=preserve_case, reduce_len=reduce_len)
        # Tokenize all tweets
        self.tokens = [tokenizer.tokenize(self.tweetFilter(tweet)) for tweet in self.tweets]

    def extractWords(self):
        # Filter tokens based on valid words
        self.words = [list(filter(self.isValidWord, tokens)) for tokens in self.tokens]
        # Create words set from words list flatten

    def tweetFilter(self, tweet):
        # Regex for valid twitter name
        twitter_name_re = "@(\w){1,15}"
        # Regex for retweet pattern
        retweet_re = "RT\s" + twitter_name_re + ":"
        # Remove retweet tokens
        no_retweet = re.sub(retweet_re, '', tweet)
        # Remove tagged users
        filtered = re.sub(twitter_name_re, '', no_retweet)
        # Return tweet filtered
        return filtered

    def isValidWord(self, word):
        # Consider word only letters and ' character
        valid_word_re = "[a-zA-Z']+"
        return re.fullmatch(valid_word_re, word)

    def __str__(self, limit=-1):
        str_obj = "%%%% TOKENIZER %%%%\n"
        for index, (tweet, tokens, words) in enumerate(zip(self.tweets, self.tokens, self.words)):
            # If limit is reached, stop print tweets
            if index == limit:
                break
            # Print all tokenization steps
            out_string = "---- Tweet N. {} ----\n" \
                         "Original\t>>> {}\n" \
                         "Tokenized\t>>> {}\n" \
                         "Valid words\t>>> {}"
            str_obj += out_string.format(index, tweet, tokens, words) + "\n"
        return str_obj + "\n"
