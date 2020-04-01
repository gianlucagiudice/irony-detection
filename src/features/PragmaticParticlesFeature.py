import re

import numpy as np

from src.Config import EMOTICONS_PATH, INITIALISM_PATH, ONOMATOPOEIC_PATH
from src.features.Debugger import Debugger
from src.features.Feature import Feature

targetPunctuation = [',', '!', '?']


def readFile(path):
    with open(path) as file:
        return [line.strip().split('\t') for line in file.readlines()]


class PragmaticParticlesFeature(Feature, Debugger):

    def __init__(self, tweets):
        super().__init__()
        # Tweets
        self.tweets = tweets
        # Features
        self.emoticonFeatureList = []
        self.initialismFeatureList = []
        self.onomatopoeicFeatureList = []
        self.punctuationFeatureList = []
        self.featuresList = []
        # Dataset
        self.emoticonsDict = {}
        self.initialismDict = {}
        self.onomatopoeicList = []

    def evaluatePragmaticParticles(self, debug=False):
        # Read dataset
        self.readDataset()
        # Evaluate emoticons
        self.evaluateEmoticons()
        # Evaluate initialism
        self.evaluateInitialism()
        # Count onomatopoeic
        self.evaluateOnomatopoeic()
        # Count punctuation
        self.evaluatePunctuation()
        # Build matrix
        self.buildMatrix(len(self.tweets), sum([len(l[0]) for l in self.featuresList]))
        # Fill matrix
        self.fillMatrix()
        # Debug info
        self.printDebugInfo(debug)
        # Return matrix
        return self.matrix

    def readDataset(self):
        self.emoticonsDict = {line[0].lower(): line[-1] for line in readFile(EMOTICONS_PATH)}
        self.initialismDict = {line[0].lower(): line[-1] for line in readFile(INITIALISM_PATH)}
        self.onomatopoeicList = [word[0].lower() for word in readFile(ONOMATOPOEIC_PATH)]

    def evaluateEmoticons(self):
        regexp = '(({}))'
        # Evaluate features
        self.emoticonFeatureList = self.evaluateFeature(self.emoticonsDict, regexp)
        self.featuresList.append(self.emoticonFeatureList)

    def evaluateInitialism(self):
        regexp = '(?=[^\w](({})+)([^\w]|$))'
        # Evaluate features
        self.initialismFeatureList = self.evaluateFeature(self.initialismDict, regexp)
        self.featuresList.append(self.initialismFeatureList)

    def evaluateOnomatopoeic(self):
        regexp = '(?=[^\w](({})+)([^\w]|$))'
        # Create auxiliary dict
        aux_dict = {key: "0" for key in self.onomatopoeicList}
        # Evaluate features
        self.onomatopoeicFeatureList = [[x[0]] for x in self.evaluateFeature(aux_dict, regexp)]
        self.featuresList.append(self.onomatopoeicFeatureList)

    def evaluateFeature(self, dictionary, regex):
        feature_list = []
        for tweet in self.tweets:
            feature = []
            for key, value in dictionary.items():
                matches = re.findall(regex.format(re.escape(key)), tweet.lower())
                # Evaluate weight of each match
                for match, *_ in matches:
                    feature += [value] * len(re.findall(re.escape(key), match))
            feature_list.append([feature.count("0"), feature.count("1")])
        return feature_list

    def evaluatePunctuation(self):
        for tweet in self.tweets:
            self.punctuationFeatureList.append({p: tweet.count(p) for p in targetPunctuation})
        self.featuresList.append([list(d.values()) for d in self.punctuationFeatureList])

    def fillMatrix(self):
        # Create matrix
        matrix = np.array(self.featuresList).transpose()
        # Flatten matrix
        self.matrix = np.array([np.concatenate(row) for row in matrix])

    def __str__(self, **kwargs):
        title = "pragmatic particles"
        header = "Tweet"
        template = "Original\t>>> \"{}\"\n"\
                   "Emot (-, +)\t>>> {}\n" \
                   "Init (-, +)\t>>> {}\n" \
                   "Onom (#)\t>>> {}\n" \
                   "Punct\t\t>>> {}"
        return super().__str__(self, self.tweets,
                               self.emoticonFeatureList, self.initialismFeatureList,
                               self.onomatopoeicFeatureList, self.punctuationFeatureList,
                               title=title, header=header, template=template)
