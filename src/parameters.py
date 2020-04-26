from .features.text import Bow, Bert

# ---- Target dataset ----
TARGET_DATASET = 'Test'
TARGET_DATASET = 'TwReyes2013'

# ---- Text feature strategy ----
TARGET_TEXT_FEATURE = 'bert'
TARGET_TEXT_FEATURE = 'bow'

if TARGET_TEXT_FEATURE == 'bow':
    TEXT_FEATURE_STRATEGY = Bow.Bow
elif TARGET_TEXT_FEATURE == 'bert':
    TEXT_FEATURE_STRATEGY = Bert.Bert

# ---- Verbose ----
DEBUG = False
