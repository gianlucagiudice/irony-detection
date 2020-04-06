import subprocess
import tempfile

from src.Config import SCRIPT_PATH, RAM_SIZE, THREAD_NUMBER
from src.features.Debugger import Debugger
from src.features.Feature import Feature

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


def build_command():
    return ['java', '-XX:ParallelGCThreads=' + str(THREAD_NUMBER), '-Xmx' + RAM_SIZE, '-jar', SCRIPT_PATH]


class PosFeature(Feature, Debugger):

    def __init__(self, tweets):
        super().__init__()
        self.tweets = tweets
        self.tagsList = []
        self.occurrenceDictList = []

    def compute_pos_tags(self, debug=False):
        # Build command for execute POS tag script
        command = build_command()
        # Get output of script
        pos_out = self.execute_pos_tagger(command).stdout
        # Parse script output
        self.parse_pos_tags(pos_out)
        # Counts tags per tweet
        self.count_tags_occurence()
        # Build matrix
        self.build_matrix(len(self.tweets), len(targetTags))
        # Fill matrix
        self.fill_matrix()
        # Debug info
        self.print_debug_info(debug)
        # Return matrix
        return self.matrix

    def execute_pos_tagger(self, command):
        # Create temp file
        with tempfile.NamedTemporaryFile() as temp_file:
            a = '\n'.join(self.tweets)
            temp_file.write(a.encode())
            temp_file.seek(0)
            # Execute script
            return subprocess.run(command + [temp_file.name], capture_output=True, text=True, check=True)


    def parse_pos_tags(self, pos_out):
        # Evaluate each tweet
        for row in pos_out.split('\n')[:-1]:
            # Unpack tags and create a list
            self.tagsList.append(row.upper().split('\t')[1].split())

    def count_tags_occurence(self):
        for tags in self.tagsList:
            occ_dict = {tag: tags.count(tag) for tag in targetTags}
            self.occurrenceDictList.append(occ_dict)

    def fill_matrix(self):
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
