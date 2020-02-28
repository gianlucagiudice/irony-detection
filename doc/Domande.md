# Domande

## Tokenizer
- Devo togliere i Retweet --> "RT @ ..."?
- Devo togliere i tag?
- Devo tenere i link?
- Come tratto le forme contratte?
    - ES. Won't
        - Ora le tengo così come sono
- RT @DjBlack_Pearl: wat muhfuckaz wearin 4 the lingerie party?????
  - wearing 4. Come tratto questo caso?

## Features
- I due paper usano features diverse.
- Table II (19:10), Lista features implementate
- Emoticons -> Conto quante sono positive e quante negative

## Pragmatic particles
- OMG = Positive/Negtive ?
- Emoji
http://kt.ijs.si/data/Emoji_sentiment_ranking/
- Considero emoticon solo se "isolata"?
- Più onomatopee attaccate tra loro?
    - ahah = 1
    - ahahahah = 2

## Lessici
- Dato che alcune sfere emotive sono presenti in più lessici "JOY" e mi aspetto che
i lessici siano coerenti tra loro, è necessario normalizzare i valori in base al 
numero di lessici in cui è presente una sfer emotiva per non far prevalere soltanto
quelli che sono presenti in tutti i lessici?
    - Se si, come normalizzo?
    
        frequenzaTot/numeroLessici?
        
        Non rischio di penalizzare troppo e avvantaggiare troppo i le sfere "uniche"?
- Non ho considerato termini d'uso e condizioni per usare ogni lessico
- EmoSenticNet alcuni lessici non sono unigrammi
- Escludo o tengo i campi "Positve" e "Negative" in EmoLex?

## Altro
- Devo normalizzare le features?
- Devo rimuovere outliers? Come rimuovo outliers quando ho una feature a più dimensione?
- Come funziona POS tagger? E' stata definita una grammatica context free?