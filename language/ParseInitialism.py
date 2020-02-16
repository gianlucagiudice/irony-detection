import re

scriptPath = "data/"
filename = "initialism.txt"
fullPath = scriptPath + filename

class ParseInitialism:

    def __init__(self, tweets):
        self.tweets = tweets
        self.initialismDict = dict()
        self.tweetsParsed = []

    def parseTweets(self):
        self.readInitialism()
        self.substituteInitialism()

    def readInitialism(self):
        with open(fullPath) as file:
            self.initialismDict = {line.split()[0]: ' '.join(line.split()[1:]) for line in file.readlines()}

    def substituteInitialism(self):
        for tweet in self.tweets:
            tweet_parsed = tweet
            for init in self.initialismDict.keys():
                tweet_parsed = re.sub(init, self.initialismDict[init], tweet_parsed)
            self.tweetsParsed.append(tweet_parsed)

    def __str__(self):
        str_obj = "%%%% PARSE INITIALISM %%%%\n"
        for row, (tweet, parsed) in enumerate(zip(self.tweets, self.tweetsParsed)):
            out_string = "---- Tweet N. {} ----\n" \
                         "Original\t>>> {}\n" \
                         "Parsed\t\t>>> {}\n"
            str_obj += out_string.format(row, tweet, parsed)
        return str_obj + "\n"
