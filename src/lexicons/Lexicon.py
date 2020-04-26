class Lexicon:
    def __init__(self, parse_strategy):
        # Parse strategy
        self.parse_strategy = parse_strategy
        # Lexicon text_feature_name
        self.name = self.compute_name()
        # Parse lexicon
        self.dict_list = self.parse()

    def compute_name(self):
        lexicon_name = self.parse_strategy.path.split('/')[-1].split('.')[0].split('-')[:-1]
        return ''.join(map(lambda x: x[0].upper() + x[1:], lexicon_name))

    def parse(self):
        return self.parse_strategy.parse()

    def evaluate_word(self, word):
        return self.dict_list.get(word, {e: 0 for e in self.emotions()})

    def emotions(self):
        return list(self.dict_list[next(iter(self.dict_list))].keys())

    def __str__(self):
        return self.name
