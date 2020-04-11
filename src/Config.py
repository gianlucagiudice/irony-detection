import os
#  ----- Script Parameters -----
TARGET_DATASET = 'Test'

# ----- Script Parameters -----
script_rel_path = "lib/"
script_name = "ark-tweet-nlp-0.3.2.jar"
SCRIPT_PATH = script_rel_path + script_name
THREAD_NUMBER = os.cpu_count()
RAM_SIZE = "2G"

# ----- Features files names -----
FILES_REL_PATH = 'data/external/features/'
# Emoticons
emoticons_file_name = 'emoticons.list'
EMOTICONS_PATH = FILES_REL_PATH + emoticons_file_name
# Initialism
initialism_file_name = 'initialism.list'
INITIALISM_PATH = FILES_REL_PATH + initialism_file_name
# Onomatopoeic
onomatopoeicFileName = 'onomatopoeic.list'
ONOMATOPOEIC_PATH = FILES_REL_PATH + onomatopoeicFileName

# ----- Lexicons file -----
lexicons_rel_path = 'data/external/lexicons/'
EMOLEX_PATH = lexicons_rel_path + 'emo-lex-v0.92.txt'
EMOSENTICNET_PATH = lexicons_rel_path + 'emo-sentic-net-v1.csv'

# ----- Dataset -----
DATASET_LABEL_NAME = '_labels'
DATASET_PATH_IN = 'data/raw/'
DATASET_PATH_OUT = 'data/processed/'

# ----- Models -----
MODELS_PATH = 'models/'

# ----- Report -----
REPORTS_PATH = 'reports/'
