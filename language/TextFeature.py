from language.Feature import Feature
from language.Debugger import Debugger

class TextFeature(Feature, Debugger):

    def __init__(self, words_list):
        super().__init__()
        self.wordsList = words_list
        self.wordsSet = None
        self.uniqueWordsList = None

    def extractTermsMatrix(self):
        # Create set of words
        self.createWordsSet()
        # Build matrix
        self.buildMatrix((len(self.wordsList)), len(self.wordsSet))
        # Fill matrix
        self.fillMatrix()

    def createWordsSet(self):
        # Create set of words
        self.wordsSet = set([word for words in self.wordsList for word in words])
        # Build list of unique words
        self.uniqueWordsList = list(self.wordsSet)

    def fillMatrix(self):
        for r, words in enumerate(self.wordsList):
            tweet_words_set = set(words)
            for c, word in enumerate(self.uniqueWordsList):
                self.matrix[r, c] += word in tweet_words_set

    def __str__(self, **kwargs):
        title = "text feature"
        header = "Matrix row"
        template = "Uniq. words >>> {}\n" \
                   "Tweet words\t>>> {}\n" \
                   "Row\t\t\t>>> {}"
        return super().__str__(self, [self.uniqueWordsList] * len(self.matrix), self.wordsList, self.matrix,
                               title=title, header=header, template=template)
