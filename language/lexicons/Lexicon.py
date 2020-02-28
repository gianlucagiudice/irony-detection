class Lexicon:
    def __init__(self, name, parseStrategy):
        # Lexicon name
        self.name = name
        # Parse strategy
        self.parseStrategy = parseStrategy
        # Parse lexicon
        self.dictList = self.parse()

    def parse(self):
        return self.parseStrategy.parse()

    def evaluateWord(self, word):
        return self.dictList.get(word,  {e: 0 for e in self.emotions()})

    def emotions(self):
        return list(self.dictList[next(iter(self.dictList))].keys())

    def __str__(self):
        return self.name
