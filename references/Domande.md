# Domande
### Tokenizer
Tante parole non esistono
### Lemmatizer
Porta solo da plurale a singolare
### Tag
Ci sono dei tweets (4) che sono etichettati come ironici ma sono semplicemente dei link
### Dataset
Ho il repo pubblico. E' un problema tenere anche il dataset nel repo?
### Altro
- Devo normalizzare le features?
- Devo rimuovere outliers? Come rimuovo outliers quando ho una feature a più dimensione?
- Come funziona POS tagger? E' stata definita una CFG?

### Riduzione matrice
31881 -> 317000 -> 28861

### Nuove
- Ho usato stem anzichè lemmatize

- Ho filtrato le words del vocabolario con un threshold di ocorrenza > 10. Tanto ho visto che con un decision tree non cambiano le cose

- Il training ci ha messo più di 24 ore perchè ho fatto 10 fold su 3 features -> 2^3 * 3 * 10

- Quando si fa k-fold cross validation, quale modello si deve tenere tra i k trainati?

- Ho solo fatto uno shuffle del dataset ma non faccio il training bilanciato

- Ho considerato 10 fold e ho parallelizzato sul numero di fold. Esiste un modo per fare il training su una porzione di dataset e parallelizazare su ogni porzione per poi "unire" la conoscenza di ogni modello trainato? Dato che su AWS macchine a 96 core
