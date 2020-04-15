import re

import numpy as np

from src.config import EMOTICONS_PATH, INITIALISM_PATH, ONOMATOPOEIC_PATH
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
        self.emoticon_feature_list = []
        self.initialism_feature_list = []
        self.onomatopoeic_feature_list = []
        self.punctuation_feature_list = []
        self.features_list = []
        # Dataset
        self.emoticons_dict = {}
        self.initialism_dict = {}
        self.onomatopoeic_list = []

    def evaluate_pragmatic_particles(self, debug=False):
        # Read dataset
        self.read_dataset()
        # Evaluate emoticons
        self.evaluate_emoticons()
        # Evaluate initialism
        self.evaluate_initialism()
        # Count onomatopoeic
        self.evaluate_onomatopoeic()
        # Count punctuation
        self.evaluate_punctuation()
        # Build matrix
        self.build_matrix(len(self.tweets), sum([len(l[0]) for l in self.features_list]))
        # Fill matrix
        self.fill_matrix()
        # Debug info
        self.print_debug_info(debug)
        # Return matrix
        return self.matrix

    def read_dataset(self):
        self.emoticons_dict = {line[0].lower(): line[-1] for line in readFile(EMOTICONS_PATH)}
        self.initialism_dict = {line[0].lower(): line[-1] for line in readFile(INITIALISM_PATH)}
        self.onomatopoeic_list = [word[0].lower() for word in readFile(ONOMATOPOEIC_PATH)]

    def evaluate_emoticons(self):
        regexp = '(({}))'
        # Evaluate features
        self.emoticon_feature_list = self.evaluateFeature(self.emoticons_dict, regexp)
        self.features_list.append(self.emoticon_feature_list)

    def evaluate_initialism(self):
        regexp = '(?=[^\w](({})+)([^\w]|$))'
        # Evaluate features
        self.initialism_feature_list = self.evaluateFeature(self.initialism_dict, regexp)
        self.features_list.append(self.initialism_feature_list)

    def evaluate_onomatopoeic(self):
        regexp = '(?=[^\w](({})+)([^\w]|$))'
        # Create auxiliary dict
        aux_dict = {key: "0" for key in self.onomatopoeic_list}
        # Evaluate features
        self.onomatopoeic_feature_list = [[x[0]] for x in self.evaluateFeature(aux_dict, regexp)]
        self.features_list.append(self.onomatopoeic_feature_list)

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

    def evaluate_punctuation(self):
        for tweet in self.tweets:
            self.punctuation_feature_list.append({p: tweet.count(p) for p in targetPunctuation})
        self.features_list.append([list(d.values()) for d in self.punctuation_feature_list])

    def fill_matrix(self):
        # Create matrix
        matrix = np.array(self.features_list).transpose()
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
                               self.emoticon_feature_list, self.initialism_feature_list,
                               self.onomatopoeic_feature_list, self.punctuation_feature_list,
                               title=title, header=header, template=template)
