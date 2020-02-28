from abc import abstractmethod
import numpy as np


class Feature:

    def __init__(self):
        self.matrix = None

    def buildMatrix(self, r, c):
        self.matrix = np.zeros((r, c))

    @abstractmethod
    def fillMatrix(self):
        return
