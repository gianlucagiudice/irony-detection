from language.lexicons.parser.ParseStrategy import ParseStrategy


class EmoLexParser(ParseStrategy):

    def parse(self):
        # Create dictionary
        dictionary = dict()
        # Parse file content
        for line in self.fileContent:
            term, emotion, value = line.split('\t')
            emotions = dictionary.get(term, dict())
            emotions.update([(emotion, int(value))])
            dictionary[term] = emotions
        return dictionary