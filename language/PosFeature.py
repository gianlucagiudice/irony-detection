from language.Feature import Feature
from language.Debugger import Debugger
from language.Config import SCRIPT_PATH, RAM_SIZE, THREAD_NUMBER
import subprocess


# Target Tags
targetTagsDescription = \
    {"N": "common noun",
     "O": "pronoun (personal; not possessive)",
     "^": "proper noun",
     "S": "nominal + possessive",
     "Z": "proper noun + possessive",
     "V": "verb",
     "A": "adjective",
     "R": "adverb",
     "!": "interjection",
     "D": "determiner",
     "P": "pre- or postposition, or subordinating conjunction",
     "&": "coordinating conjunction",
     "T": "verb particle",
     "X": "existential there",
     "L": "nominal + verbal (e. g., i’m)",
     "M": "proper noun + verbal",
     "Y": "X + verbal"}
targetTags = targetTagsDescription.keys()


def buildCommand():
    return ['java', '-XX:ParallelGCThreads=' + THREAD_NUMBER, '-Xmx' + RAM_SIZE, '-jar', SCRIPT_PATH]


class PosFeature(Feature, Debugger):

    def __init__(self, tweets):
        super().__init__()
        self.matrix = None
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
        self.buildMatrix(len(self.tweets), len(targetTags))
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

    def fillMatrix(self):
        for r, _ in enumerate(self.tweets):
            self.matrix[r, :] = list(self.occurrenceDictList[r].values())

    def __str__(self, **kwargs):
        title = "pos tagger"
        header = "Tweet"
        template = "Original\t>>> \"{}\"\n"\
                   "Tags\t\t>>> {}\n" \
                   "Occ.\t\t>>> {}"
        return super().__str__(self, self.tweets, self.tagsList, self.occurrenceDictList,
                               title=title, header=header, template=template)
