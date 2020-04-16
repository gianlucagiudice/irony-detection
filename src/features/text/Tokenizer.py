import re

from nltk.corpus import stopwords
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
        # Words occurrences
        self.word_occurrence = dict()

    def tokenize(self):
        print("> Tokenizer . . .")
        # Extract valid words
        self.extract_words()
        # Remove words below threshold
        self.words_list = self.filter_below_threshold(self.words_list)
        # Return list of words
        return self.words_list

    def extract_words(self):
        # Construct tokenizer object
        tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
        # Stemmer
        stemmer = PorterStemmer()
        # Start tokenizing
        print('\t{}% completed'.format(0), end='')
        # Tokenize all tweets
        for index, tweet in enumerate(self.tweets_list, 1):
            tokens = tokenizer.tokenize(tweet)
            words = [token for token in tokens if self.is_valid_word(token)]
            words_stemmed = self.stem_words(words, stemmer)
            self.update_occurrences(words_stemmed)
            self.words_list.append(words_stemmed)
            # Progress
            if index % 1000 == 0:
                print('\r\t{}% completed'.format(round(index / len(self.tweets_list) * 100, 3)), end='')
        print('\r\t{}% tokenized'.format(100))

    def filter_below_threshold(self, words_list):
        filtered = []
        for words in words_list:
            filtered.append([word for word in words if self.word_occurrence[word] > OCCURRENCE_THRESHOLD])
        return filtered

    def update_occurrences(self, words):
        for word in words:
            self.word_occurrence[word] = self.word_occurrence.get(word, 0) + 1

    def is_valid_word(self, token):
        valid_word_re = "[a-zA-Z]+"
        return re.fullmatch(valid_word_re, token) and token not in self.stopwords and token != 'rt'

    def stem_words(self, words, stemmer):
        return [stemmer.stem(word) for word in words]

    def __str__(self, **kwargs):
        title = "tokenizer"
        header = "Tweet"
        template = "Original\t>>> \"{}\"\n"\
                   "Tokenized\t>>> {}\n" \
                   "Valid words\t>>> {}"
        return super().__str__(self, self.tweets_list, self.tokens_list, self.words_list,
                               title=title, header=header, template=template)
