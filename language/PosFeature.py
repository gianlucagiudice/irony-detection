'''
        LIST OF POS FEATURE
"N":"common noun", "O":"pronoun (personal; not possessive)", "ˆ":"proper noun", "S":"nominal + possessive", "Z":"proper noun + possessive", "V":"verb", "A":"adjective", "R":"adverb", "!":"interjection", "D":"determiner", "P":"pre- or postposition, or subordinating conjunction", "&":"coordinating conjunction", "T":"verb particle", "X":"existential there", "L":"nominal + verbal (e. g., i’m)", "M":"proper noun + verbal", "Y":"X + verbal"
'''
import subprocess
import re

# Script Parameters
scriptPath = "language/"
scriptName = "ark-tweet-nlp-0.3.2.jar"
fullPath = scriptPath + scriptName
threadNumber = "4"
ramSize = "1G"

# Target tags
targetTags = {"N": "common noun", "O": "pronoun (personal; not possessive)", "^": "proper noun",
              "S": "nominal + possessive", "Z": "proper noun + possessive", "V": "verb", "A":"adjective",
              "R": "adverb", "!": "interjection", "D": "determiner",
              "P": "pre- or postposition, or subordinating conjunction", "&": "coordinating conjunction",
              "T": "verb particle", "X": "existential there", "L": "nominal + verbal (e. g., i’m)",
              "M": "proper noun + verbal", "Y": "X + verbal"}


def buildCommand():
    return ['java', '-XX:ParallelGCThreads=' + threadNumber, '-Xmx' + ramSize, '-jar', fullPath]


class PosFeature:

    def __init__(self, words, tweets):
        self.targetTags = targetTags
        self.words = words
        self.tweets = tweets
        self.dictsList = []
        self.tags = []

    def computePosTag(self):
        # Build command for execute POS tag script
        command = buildCommand()
        # Get output of script
        pos_out = self.executePosTagger(command).stdout
        # Parse script output
        self.parsePosTag(pos_out)
        # Extract all tags from each tweet
        all_tags = self.extractTags()
        # Filter tags based on set of target tags
        self.tagFilter(all_tags)

    def executePosTagger(self, command):
        # Set stdin equal to tweets separated by newline
        input_tweets = ''.join(self.tweets)
        # Execute script
        return subprocess.run(command, capture_output=True, text=True, check=True, input=input_tweets)

    def parsePosTag(self, pos_out):
        # Evaluate each tweet
        for row in pos_out.split('\n')[:-1]:
            # Unpack tokens and tags converted to lowercase
            tokens, tags, *_ = row.lower().split('\t')
            # Create per row dictionary token : tag
            tweet_dict = dict()
            for token, tag in zip(tokens.split(' '), tags.split(' ')):
                # Finds all words in script-token. Must be coherent with custom tokenizer
                tweet_dict.update([(partial_token, tag) for partial_token in re.findall("\w+", token)])
            self.dictsList.append(tweet_dict)

    def extractTags(self):
        # Consider the pair dictionary - words associated to each tweet
        tweets = zip(self.words, self.dictsList)
        # Return list og tags for each tweet
        return [[tokens_dict[word].upper() for word in words] for words, tokens_dict in tweets]

    def tagFilter(self, tags_list):
        # Get set of valid tags
        valid_tags = set(self.targetTags.keys())
        # Filter tags based on valid tags
        for tags in tags_list:
            self.tags.append([tag for tag in tags if tag in valid_tags])

    def __str__(self, limit=-1):
        str_obj = "%%%% POS TAGGER %%%%\n"
        for index, (tweet, words, tags) in enumerate(zip(self.tweets, self.words, self.tags)):
            # If limit is reached, stop print tweets
            if index == limit:
                break
            # Print all tokenization steps
            out_string = "---- Tweet N. {} ----\n" \
                         "Original\t>>> {}\n" \
                         "Words\t\t>>> {}\n" \
                         "Tags\t\t>>> {}"
            tags_description = [self.targetTags[tag] for tag in tags]
            str_obj += out_string.format(index, tweet.strip(), words, tags_description) + "\n"
        return str_obj + "\n"