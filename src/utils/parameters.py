import sys

from src.features.text import Bow, Bert, Sbert

# ---- Target dataset ----
TARGET_DATASET = None

# ---- Target text feature ----
TARGET_TEXT_FEATURE = None

# ---- Text feature strategy ----
TEXT_FEATURE_STRATEGY = None


def parse_parameters_main(argv):
    if len(argv) != 3:
        print("Error: Invalid parameters!")
        quit(-1)
    target_dataset = argv[1]
    target_text_feature = argv[2]

    text_feature_strategy = None
    if target_text_feature == 'bow':
        text_feature_strategy = Bow.Bow
    elif target_text_feature == 'bert':
        text_feature_strategy = Bert.Bert
    elif target_text_feature == 'sbert':
        text_feature_strategy = Sbert.Sbert
    else:
        print("Error: Invalid text feature!")
        quit(-2)
    return target_dataset, target_text_feature, text_feature_strategy


def parse_parameters_training(argv):
    if len(argv) != 2:
        print("Error: Invalid parameters!")
        quit(-1)
    return argv[1]


script_name = sys.argv[0].split('/')[-1]
if script_name == 'main.py':
    TARGET_DATASET, TARGET_TEXT_FEATURE, TEXT_FEATURE_STRATEGY = parse_parameters_main(sys.argv)
elif script_name == 'training.py' or script_name == 'pca.py':
    TARGET_DATASET = parse_parameters_training(sys.argv)
