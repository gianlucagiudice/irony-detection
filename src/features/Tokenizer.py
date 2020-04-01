import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

from src.features.Debugger import Debugger


class Tokenizer(Debugger):
    def __init__(self, tweets):
        # List of tweets
        self.tweetsList = tweets
        # List of tweets tokenized
        self.tokensList = []
        # List of words in each tweet
        self.wordsList = []
        # Set of stopwords
        self.stopwords = set(stopwords.words('english'))
        # Lemmatizer
        self.lemmatizer = WordNetLemmatizer()

    def parseTweets(self, debug=False):
        # Tokenize tweets
        self.tokenizeTweets()
        # Extract valid words
        self.extractWords()
        # Debug info
        self.printDebugInfo(debug)
        # Return list of words
        return self.wordsList

    def tokenizeTweets(self):
        # Construct tokenizer object
        preserve_case = False  # Convert each work to lowercase
        reduce_len = True  # mooooooonkey -> mooonkey
        tokenizer = TweetTokenizer(preserve_case=preserve_case, reduce_len=reduce_len)
        # Tokenize all tweets
        self.tokensList = [tokenizer.tokenize(self.tweetFilter(tweet)) for tweet in self.tweetsList]

    def extractWords(self):
        # Filter tokens based on valid words
        for tokens in self.tokensList:
            words = list(filter(self.isValidWord, tokens))
            self.wordsList.append(self.lemmatize(words))

    def tweetFilter(self, tweet):
        # Regexp for valid twitter name
        twitter_name_re = "@(\w){1,15}"
        # Regexp for retweet pattern
        retweet_re = "RT\s" + twitter_name_re + ":"
        # Remove retweet tokens
        no_retweet = re.sub(retweet_re, '', tweet)
        # Remove tagged users
        filtered = re.sub(twitter_name_re, '', no_retweet)
        # Return tweet filtered
        return filtered

    def isValidWord(self, word):
        # Consider word only letters and ' character
        valid_word_re = "[a-zA-Z]+"
        return re.fullmatch(valid_word_re, word) and word not in self.stopwords

    def lemmatize(self, words):
        return [self.lemmatizer.lemmatize(word) for word in words]

    def __str__(self, **kwargs):
        title = "tokenizer"
        header = "Tweet"
        template = "Original\t>>> \"{}\"\n"\
                   "Tokenized\t>>> {}\n" \
                   "Valid words\t>>> {}"
        return super().__str__(self, self.tweetsList, self.tokensList, self.wordsList,
                               title=title, header=header, template=template)
