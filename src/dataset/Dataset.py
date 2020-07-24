import json

from src.utils.config import DATASET_LABEL_NAME
from src.utils.config import DATASET_PATH_IN


class Dataset:
	def __init__(self, dataset_name):
		self.dataset_name = dataset_name
		self.ironic_dict = None
		self.tweets = []
		self.labels = []

	def extract(self, target_dataset=None):
		# Target dataset (ironic/non_ironic)
		if target_dataset is None:
			target_dataset = {'ironic': True, 'non_ironic': True}
		# Extract dict containing ironic tweets files
		self.extract_dict(self.dataset_name)
		# Read tweets
		target_dict = self.build_target_dict(target_dataset)
		self.read_tweets(self.dataset_name, target_dict)
		# Return data
		return self.tweets

	def extract_dict(self, dataset_name):
		path = '{}{}/{}.json'.format(DATASET_PATH_IN, dataset_name, DATASET_LABEL_NAME)
		with open(path) as json_dict:
			self.ironic_dict = json.load(json_dict)

	def read_tweets(self, dataset_name, file_dict):
		for file_name, label in file_dict.items():
			# Read all tweets in file
			path = '{}{}/{}'.format(DATASET_PATH_IN, dataset_name, file_name)
			with open(path) as file:
				content = [tweet.strip() for tweet in file.readlines()]
				self.tweets += content
				self.labels += [label] * len(content)

	def build_target_dict(self, type):
		return {key: value for key, value in self.ironic_dict.items() if type[value]}
