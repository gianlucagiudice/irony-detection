args=("$@")
python3 main.py ${args[0]} bow
python3 main.py ${args[0]} bert
python3 main.py ${args[0]} sbert
python3 training.py ${args[0]}
python3 pca.py ${args[0]}
