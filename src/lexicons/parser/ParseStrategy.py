from abc import abstractmethod


class ParseStrategy:
    def __init__(self, path):
        self.path = path
        self.fileContent = self.readFile()

    def readFile(self):
        with open(self.path) as file:
            return [line.strip() for line in file.readlines()]

    @abstractmethod
    def parse(self):
        return
