# ----- Script Parameters -----
script_rel_path = "lib/"
script_name = "ark-tweet-nlp-0.3.2.jar"
SCRIPT_PATH = script_rel_path + script_name
THREAD_NUMBER = "4"
RAM_SIZE = "1G"

# ----- Features files names -----
files_rel_path = 'data/features/'
# Emoticons
emoticons_file_name = 'emoticons.list'
EMOTICONS_PATH = files_rel_path + emoticons_file_name
# Initialism
initialism_file_name = 'initialism.list'
INITIALISM_PATH = files_rel_path + initialism_file_name
# Onomatopoeic
onomatopoeicFileName = 'onomatopoeic.list'
ONOMATOPOEIC_PATH = files_rel_path + onomatopoeicFileName

# ----- Lexicons file -----
lexicons_rel_path = 'data/lexicons/'
EMOLEX_PATH = lexicons_rel_path + 'emo-lex-v0.92.txt'
EMOSENTICNET_PATH = lexicons_rel_path + 'emo-sentic-net-v1.csv'