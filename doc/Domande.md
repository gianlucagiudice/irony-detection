# Domande

## Tokenizer
- Devo togliere i Retweet --> "RT @ ..."?
- Devo togliere i tag?
- Devo tenere i link?
- Come tratto le forme contratte?
    - ES. Won't
        - Ora le tengo così come sono
        - ark-tweet-npl crea due token "won", "t"
- RT @DjBlack_Pearl: wat muhfuckaz wearin 4 the lingerie party?????
  - wearing 4. Come tratto questo caso?
- RT @eye_ee_duh_Esq: LMBO! This man filed an EMERGENCY Motion for Continuance on account of the Rangers game tonight! « Wow lmao
  - Come tratto "«" ?
- Deve essere case sensitive?
- Va bene applicare riduzioni delle frasi?
    - mooooooonkey >>> mooonkey
    - ???????? >>> ??? 
- Cosa viene considerato come termine?
    - La punteggiatura è considerato termine?
    - Attualmente considero validi i termini del tipo:
        - [a-zA-Z']+
        - In questo modo tengo le forme contratte
- E' un problema se ad ogni esecuzione dello script, ho una permutazione su T = {t_1, t_2, ...t_n}, dove T è la colonna della matrice
- Ho fatto il tokenizer ma ho visto che è già implementato nello script. Quale devo usare?

## Pos tag
- Devo taggare le words sul mio tokenizer?
- Se costruisco prima la matrice e poi guardo i tag assegnati, nel caso di 2 occerrenze in un tweet perdo iniettività
- POS tagger script ---> BUG?

#### BUG pos tagger?
Problema parol

》have

è considerata parola