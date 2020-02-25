from language.Feature import Feature


class TextFeature(Feature):

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

    def __str__(self):
        str_obj = "%%%% TEXT FEATURE %%%%\n"
        for r, words in enumerate(self.wordsList):
            out_string = "---- Matrix row N. {} ----\n" \
                         "Uniq. words >>> {}\n" \
                         "Tweet words\t>>> {}\n" \
                         "Row\t\t\t>>> {}"
            str_obj += out_string.format(r, self.uniqueWordsList, words, str(self.matrix[r])) + "\n"
        return str_obj + "\n"
