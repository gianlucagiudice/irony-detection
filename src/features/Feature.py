from abc import abstractmethod

import numpy as np


class Feature:

	def __init__(self):
		self.matrix = None

	def build_matrix(self, r, c):
		self.matrix = np.zeros((r, c), dtype=int)

	@abstractmethod
	def fill_matrix(self, *args):
		return
