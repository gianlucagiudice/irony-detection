import subprocess
import numpy as np

# Script Parameters
scriptPath = "lib/"
scriptName = "ark-tweet-nlp-0.3.2.jar"
fullPath = scriptPath + scriptName
threadNumber = "4"
ramSize = "1G"

# Target tags
targetTags = ['N', 'O', '^', 'S', 'Z', 'V', 'A', 'R', '!', 'D', 'P', '&', 'T', 'X', 'L', 'M', 'Y']
# Target Description
targetTagsDescription = \
    {"N": "common noun", "O": "pronoun (personal; not possessive)", "^": "proper noun",
     "S": "nominal + possessive", "Z": "proper noun + possessive", "V": "verb", "A": "adjective",
     "R": "adverb", "!": "interjection", "D": "determiner",
     "P": "pre- or postposition, or subordinating conjunction", "&": "coordinating conjunction",
     "T": "verb particle", "X": "existential there", "L": "nominal + verbal (e. g., i’m)",
     "M": "proper noun + verbal", "Y": "X + verbal"}


def buildCommand():
    return ['java', '-XX:ParallelGCThreads=' + threadNumber, '-Xmx' + ramSize, '-jar', fullPath]


class PosFeature:

    def __init__(self, tweets):
        self.matrix = None
        self.targetTags = targetTags
        self.tweets = tweets
        self.tagsList = []
        self.occurrenceDictList = []

    def computePosTags(self):
        # Build command for execute POS tag script
        command = buildCommand()
        # Get output of script
        pos_out = self.executePosTagger(command).stdout
        # Parse script output
        self.parsePosTags(pos_out)
        # Counts tags per tweet
        self.countTagsOccurence()
        # Build matrix
        self.buildMatrix()
        # Fill matrix
        self.fillMatrix()

    def executePosTagger(self, command):
        # Set stdin equal to tweets separated by newline
        input_tweets = '\n'.join(self.tweets)
        # Execute script
        return subprocess.run(command, capture_output=True, text=True, check=True, input=input_tweets)

    def parsePosTags(self, pos_out):
        # Evaluate each tweet
        for row in pos_out.split('\n')[:-1]:
            # Unpack tags and create a list
            self.tagsList.append(row.upper().split('\t')[1].split())

    def countTagsOccurence(self):
        for tags in self.tagsList:
            occ_dict = {tag: tags.count(tag) for tag in targetTags}
            self.occurrenceDictList.append(occ_dict)

    def buildMatrix(self):
        self.matrix = np.zeros(((len(self.tweets)), len(self.targetTags)))

    def fillMatrix(self):
        for r, _ in enumerate(self.tweets):
            self.matrix[r, :] = list(self.occurrenceDictList[r].values())

    def __str__(self, limit=-1):
        str_obj = "%%%% POS TAGGER %%%%\n"
        for index, (tweet, tags, occurence_dict) in enumerate(zip(self.tweets, self.tagsList, self.occurrenceDictList)):
            # If limit is reached, stop print tweets
            if index == limit:
                break
            # Print all tokenization steps
            out_string = "---- Tweet N. {} ----\n" \
                         "Original\t>>> {}\n" \
                         "Tags\t\t>>> {}\n" \
                         "Occ.\t\t>>> {}"
            str_obj += out_string.format(index, tweet.strip(), tags, occurence_dict) + "\n"
        return str_obj + "\n"