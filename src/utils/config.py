import os

# ----- Script Parameters -----
script_rel_path = "lib/"
script_name = "ark-tweet-nlp-0.3.2.jar"
SCRIPT_PATH = script_rel_path + script_name
WEKA_PATH = '{}/weka.jar'.format(script_rel_path)
THREAD_NUMBER = os.cpu_count()
RAM_SIZE = "2G"

# ----- Features files names -----
FILES_REL_PATH = 'data/external/features/'
# Emoticons
EMOTICONS_FILE_NAME = 'emoticons.list'
EMOTICONS_PATH = FILES_REL_PATH + EMOTICONS_FILE_NAME
# Initialism
INITIALISM_FILE_NAME = 'initialism.list'
INITIALISM_PATH = FILES_REL_PATH + INITIALISM_FILE_NAME
# Onomatopoeic
ONOMATOPOEIC_FILE_NAME = 'onomatopoeic.list'
ONOMATOPOEIC_PATH = FILES_REL_PATH + ONOMATOPOEIC_FILE_NAME

# ----- Lexicons file -----
LEXICONS_REL_PATH = 'data/external/lexicons/'
EMOLEX_PATH = LEXICONS_REL_PATH + 'emo-lex-v0.92.txt'
EMOSENTICNET_PATH = LEXICONS_REL_PATH + 'emo-sentic-net-v1.csv'

# ----- Dataset -----
DATASET_LABEL_NAME = '_labels'
DATASET_PATH_IN = 'data/raw/'
DATASET_PATH_OUT = 'data/processed/'

# ----- Models -----
MODELS_PATH = 'models/'

# ----- Report -----
REPORTS_PATH = 'reports/'
