class Lexicon:
    def __init__(self, parseStrategy):
        # Parse strategy
        self.parseStrategy = parseStrategy
        # Lexicon name
        self.name = self.computeName()
        # Parse lexicon
        self.dictList = self.parse()

    def computeName(self):
        lexicon_name = self.parseStrategy.path.split('/')[-1].split('.')[0].split('-')[:-1]
        return ''.join(map(lambda x: x[0].upper() + x[1:], lexicon_name))

    def parse(self):
        return self.parseStrategy.parse()

    def evaluateWord(self, word):
        return self.dictList.get(word, {e: 0 for e in self.emotions()})

    def emotions(self):
        return list(self.dictList[next(iter(self.dictList))].keys())

    def __str__(self):
        return self.name
