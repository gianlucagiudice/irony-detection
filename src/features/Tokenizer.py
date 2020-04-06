import re

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from operator import itemgetter

from src.features.Debugger import Debugger


class Tokenizer(Debugger):
    def __init__(self, tweets, occurence_threshold=5):
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
        # Words occurences
        self.occurence_number = dict()
        # Occurences threshold
        # 10 ---> 3967
        # 5 ---> 6316
        self.occurence_threshold = occurence_threshold


    def parse_tweets(self, debug=False):
        # Tokenize tweets
        # self.tokenizeTweets()
        # Extract valid words
        self.extractWords()
        # Remove words below threashold
        self.wordsList = self.filt_below_threshold(self.wordsList)
        # Save words occurences
        self.save_occurrences()
        # Return list of words
        return self.wordsList

    def tokenizeTweets(self):
        # Construct tokenizer object
        tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
        # nltk.pos_tag(tokenizer.tokenize(self.tweetsList[37].lower()))
        # Tokenize all tweets
        self.tokensList = [tokenizer.tokenize(self.tweetFilter(tweet)) for tweet in self.tweetsList]

    def extractWords(self):
        # Construct tokenizer object
        tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
        # TODO: Eliminare indice
        for i, tweet in enumerate(self.tweetsList):
            tokens = nltk.pos_tag(tokenizer.tokenize(tweet))
            words = [token for token in tokens if self.isValidWord(token)]
            words_lemmatized = self.lemmatize(words)
            self.update_occurrences(words_lemmatized)
            self.wordsList.append(words_lemmatized)

        # Filt below threshold

    def wordFilter(self, tweet):
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

    def filt_below_threshold(self, wordsList):
        filtered = []
        for words in wordsList:
            filtered.append([word for word in words if self.occurence_number[word] > self.occurence_threshold])
        return filtered

    def update_occurrences(self, words):
        for word in words:
            self.occurence_number[word] = self.occurence_number.get(word, 0) + 1

    def isValidWord(self, token):
        word, _ = token
        # Consider word only letters and ' character
        valid_word_re = "[a-zA-Z]+"
        return re.fullmatch(valid_word_re, word) and word not in self.stopwords and word != 'rt'

    def lemmatize(self, words):
        # ret = []
        # list = [wordnet.ADJ, wordnet.VERB, wordnet.NOUN, wordnet.ADV]
        # for word, tag in words:
        #     s = set()
        #     for t in list:
        #         s.add(self.lemmatizer.lemmatize(word, t))
        #     if len(s) > 1:
        #         print(word)
        #     ret.append(s.pop())
        # return ret
        return [self.lemmatizer.lemmatize(word) for word, tag in words]

    def save_occurrences(self):
        # Write dict
        with open('words_occurence.list', 'w') as file:
            for key, value in sorted(self.occurence_number.items(), key=itemgetter(1), reverse=True):
                file.write("{},{}\n".format(key, value))

    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def __str__(self, **kwargs):
        title = "tokenizer"
        header = "Tweet"
        template = "Original\t>>> \"{}\"\n"\
                   "Tokenized\t>>> {}\n" \
                   "Valid words\t>>> {}"
        return super().__str__(self, self.tweetsList, self.tokensList, self.wordsList,
                               title=title, header=header, template=template)
