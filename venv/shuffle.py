import pandas as pd
import spacy


# nlp = spacy.load('en_core_web_sm') # Language Model
# doc = nlp(u'Apple is the the looking at buying U.K. startup for $1 billion')
# for token in doc:
#     print(token.text)
#     print(token.is_stop)

df = pd.read_csv('C:/Users/D072828/PycharmProjects/Master_Thesis/venv/train_full.csv', header=None, nrows=100000)
ds = df.sample(frac=1)

print(ds)

ds.to_csv('shuffled_.csv', index = False, header = False)


