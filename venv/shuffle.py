import pandas as pd
import spacy


# nlp = spacy.load('en_core_web_sm') # Language Model
# doc = nlp(u'Apple is the the looking at buying U.K. startup for $1 billion')
# for token in doc:
#     print(token.text)
#     print(token.is_stop)

# df = pd.read_csv('C:/Users/D072828/PycharmProjects/Master-Thesis/venv/train_full.csv', header=None, nrows=100000)
df = pd.read_csv('C:/Users/D072828/Desktop/Master Thesis/Datasets for thesis/yelp_review_full_csv/train.csv', header=None, nrows=100000)
ds = df.sample(frac=1)

print(ds)

ds.to_csv('shuffled_full_100k_v3.csv', index = False, header = False)


