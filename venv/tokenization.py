import spacy

nlp = spacy.load('en_core_web_sm') # Language Model
doc = nlp(u'Apple is the the looking at buying U.K. startup for $1 billion')
for token in doc:
    print(token.text)

print(nlp)
print(type(nlp))


from spacy.vocab import Vocab
from spacy.language import Language
nlp = Language(Vocab())
from spacy.lang.en import English
nlp = English()
# doc = nlp(u'Apple is the the looking at buying U.K. startup for $1 billion')
doc = nlp("I always love a good diner.  Gab and Eat was just wha")

print (type(doc))
print (doc)
for token in doc:
    print(token.text)
    print(token.is_stop)
    print('punct' , token.is_punct)
    print('norm' , token.norm_)
    # print(token)
print(nlp)
print(type(nlp))
