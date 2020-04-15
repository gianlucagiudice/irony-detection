# Repository class


class Tweets:
    def __init__(self, tweets):
        self.values = tweets
        self.tokens = None

    def save_tokens(self, tokens):
        self.tokens = tokens
