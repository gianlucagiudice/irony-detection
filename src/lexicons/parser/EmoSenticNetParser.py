from src.lexicons.parser.ParseStrategy import ParseStrategy


class EmoSenticNetParser(ParseStrategy):

	def parse(self):
		# Create dictionary
		dictionary = dict()
		# Get keys in csv header
		keys = [key.lower() for key in self.fileContent[0].split(',')[1:]]
		# Parse file content
		for line in self.fileContent[1:]:
			term, *values = line.split(',')
			target_values = list(map(lambda x: int(x), values))
			dictionary[term] = dict(zip(keys, target_values))
		return dictionary
