import re
from operator import itemgetter

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

from src.features.Debugger import Debugger

OCCURRENCE_THRESHOLD = 10


class Tokenizer(Debugger):
    def __init__(self, tweets):
        # List of tweets
        self.tweets_list = tweets
        # List of tweets tokenized
        self.tokens_list = []
        # List of words in each tweet
        self.words_list = []
        # Set of stopwords
        self.stopwords = set(stopwords.words('english'))
        # Stemmer
        self.stemmer = PorterStemmer()
        # Words occurrences
        self.word_occurrence = dict()

    def parse_tweets(self, debug=False):
        # Extract valid words
        self.extract_words()
        # Remove words below threashold
        self.words_list = self.filter_below_threshold(self.words_list)
        # Return list of words
        return self.words_list

    def extract_words(self):
        # Construct tokenizer object
        tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
        for tweet in self.tweets_list:
            tokens = nltk.pos_tag(tokenizer.tokenize(tweet))
            words = [token for token in tokens if self.is_valid_word(token)]
            words_stemmed = self.stem_words(words)
            self.update_occurrences(words_stemmed)
            self.words_list.append(words_stemmed)

    def filter_below_threshold(self, words_list):
        filtered = []
        for words in words_list:
            filtered.append([word for word in words if self.word_occurrence[word] > OCCURRENCE_THRESHOLD])
        return filtered

    def update_occurrences(self, words):
        for word in words:
            self.word_occurrence[word] = self.word_occurrence.get(word, 0) + 1

    def is_valid_word(self, token):
        word, _ = token
        # Consider word only letters and ' character
        valid_word_re = "[a-zA-Z]+"
        return re.fullmatch(valid_word_re, word) and word not in self.stopwords and word != 'rt'

    def stem_words(self, words):
        return [self.stemmer.stem(word) for word, tag in words]

    def save_occurrences(self):
        # Write dict
        with open('words_occurence.list', 'w') as file:
            for key, value in sorted(self.word_occurrence.items(), key=itemgetter(1), reverse=True):
                file.write("{},{}\n".format(key, value))

    def __str__(self, **kwargs):
        title = "tokenizer"
        header = "Tweet"
        template = "Original\t>>> \"{}\"\n"\
                   "Tokenized\t>>> {}\n" \
                   "Valid words\t>>> {}"
        return super().__str__(self, self.tweets_list, self.tokens_list, self.words_list,
                               title=title, header=header, template=template)
