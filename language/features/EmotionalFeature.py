import numpy as np

from language.Config import EMOLEX_PATH, EMOSENTICNET_PATH
from language.features.Debugger import Debugger
from language.features.Feature import Feature
from language.lexicons.Lexicon import Lexicon
from language.lexicons.parser.EmoLexParser import EmoLexParser
from language.lexicons.parser.EmoSenticNetParser import EmoSenticNetParser

# Lexicons list
lexiconList = [Lexicon(EmoLexParser(EMOLEX_PATH)),
               Lexicon(EmoSenticNetParser(EMOSENTICNET_PATH))]


class EmotionalFeature(Feature, Debugger):

    def __init__(self, words_list):
        super().__init__()
        self.wordsList = words_list
        # Set list of lexicons for emotional features
        self.lexicons = lexiconList
        # List of emoticons
        self.emotionalList = []
        # List of emotional feature combined foreach lexicon
        self.emotionalFeatureList = []

    def evaluateEmotions(self, debug=False):
        # Evaluate tweets using all lexicons
        self.evaluateLexicons()
        # Combine lexicons
        self.combineLexicons()
        # Build matrix
        self.buildMatrix(len(self.wordsList), len(self.emotionalFeatureList[0]))
        # Fill matrix with lexicons value
        self.fillMatrix()
        # Debug info
        self.printDebugInfo(debug)
        # Return matrix
        return self.matrix

    def evaluateLexicons(self):
        for words in self.wordsList:
            row = dict()
            for lexicon in self.lexicons:
                # List of values relative to all words
                values = [list(lexicon.evaluateWord(word).values()) for word in words]
                # Sum all values to obtain per-lexicon values
                total = np.array(values).sum(axis=0)
                # Reconstruct dictionary form matrix
                row[lexicon] = {e: int(v) for e, v in zip(lexicon.emotions(), total)}
            # Add target lexicon to specific list of words
            self.emotionalList.append(row)

    def combineLexicons(self):
        # Set of emotions to exclude
        exclude_emotions = {"positive", "negative"}
        all_emotions = set([e for lexicon in self.lexicons for e in lexicon.emotions()])
        # Compute emotions to combine
        target_emotions = all_emotions - exclude_emotions
        keys = list(target_emotions)
        # Iterate over each tweet in order to combine each lexicon
        for lexicons in self.emotionalList:
            # Initialize combined dictionary to 0s
            combined = {k: 0 for k in keys}
            # Iterate and combine each lexicon
            for lexicon in lexicons.values():
                for k, v in lexicon.items():
                    if k in target_emotions:
                        combined[k] = combined[k] + v
            # Append the new combined lexicon
            self.emotionalFeatureList.append(combined)

    def fillMatrix(self):
        for r, lexicon in enumerate(self.emotionalFeatureList):
            self.matrix[r:] = list(lexicon.values())

    def __str__(self, **kwargs):
        title = "emotional feature"
        header = "Tweet"
        template = "Words\t\t >>> \"{}\"\n" + \
                   "{:<13}>>> {}\n" * len(self.lexicons) + \
                   "Combined\t >>> {}"
        # Build auxiliary list in order to print per-lexicon debug info
        aux_debug: [list] = []
        for lexicon in self.lexicons:
            aux_debug += [[str(lexicon)] * len(self.wordsList)] + \
                         [[tweet[lexicon] for tweet in self.emotionalList]]
        # Return debug info
        return super().__str__(self, self.wordsList, *aux_debug, self.emotionalFeatureList,
                               title=title, header=header, template=template)