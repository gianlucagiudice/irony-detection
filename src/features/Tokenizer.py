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
        self.tweets_list = tweets
        # List of tweets tokenized
        self.tokens_list = []
        # List of words in each tweet
        self.words_list = []
        # Set of stopwords
        self.stopwords = set(stopwords.words('english'))
        # Lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        # Words occurences
        self.occurence_number = dict()
        # Occurences threshold
        self.occurence_threshold = occurence_threshold


    def parse_tweets(self, debug=False):
        # Extract valid words
        self.extract_words()
        # Remove words below threashold
        self.words_list = self.filt_below_threshold(self.words_list)
        # Return list of words
        return self.words_list

    def extract_words(self):
        # Construct tokenizer object
        tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
        for tweet in self.tweets_list:
            tokens = nltk.pos_tag(tokenizer.tokenize(tweet))
            words = [token for token in tokens if self.is_valid_word(token)]
            words_lemmatized = self.lemmatize(words)
            self.update_occurrences(words_lemmatized)
            self.words_list.append(words_lemmatized)

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

    def filt_below_threshold(self, words_list):
        filtered = []
        for words in words_list:
            filtered.append([word for word in words if self.occurence_number[word] > self.occurence_threshold])
        return filtered

    def update_occurrences(self, words):
        for word in words:
            self.occurence_number[word] = self.occurence_number.get(word, 0) + 1

    def is_valid_word(self, token):
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
        return super().__str__(self, self.tweets_list, self.tokens_list, self.words_list,
                               title=title, header=header, template=template)
