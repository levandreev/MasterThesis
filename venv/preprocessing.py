import pandas as pd
import spacy
import csv
import itertools

corpus = []
y = []
N = 1000

nlp = spacy.load('en_core_web_sm') # Language Model
#
# df = pd.read_csv('C:/Users/D072828/PycharmProjects/Thesis/venv/train_full.csv', header=None, nrows=100000)
# ds = df.sample(frac=1)
# print(ds)
# ds.to_csv('C:/Users/D072828/PycharmProjects/Thesis/venv/to_be_preprocessed_100_k.csv', index = False, header = False)

with open('C:/Users/D072828/PycharmProjects/Master_Thesis/venv/to_be_preprocessed_10_k.csv', newline='') as csvfile:
    yelp = csv.reader(csvfile, delimiter=',')
    for row in itertools.islice(yelp, N):
         clean_row = row[1].strip().replace('"','').replace(';','')
         # clean_row = row[0].strip().replace('"','').replace(';','')
         target_class = row[0]
         # target_class = clean_row[0]
         corpus.append(clean_row)
         # corpus.append(clean_row[2:])
         y.append(int(target_class))

# print(y)
# print(corpus[:2])
# print(len(corpus)==len(y))

preprocessed_corpus = []
for row in corpus:
    doc = nlp(row)
    preprocessed_row = ""
    for token in doc:
        if not token.is_punct and token.is_alpha:
            # preprocessed_row.append(token.lemma_)
            preprocessed_row+= (token.lemma_ + ' ')
            # print(preprocessed_row)
    preprocessed_corpus.append(preprocessed_row)
    # print(preprocessed_corpus)
    # print(doc)
    # preprocessed_corpus.append(doc)


# print(len(preprocessed_corpus) == len(y))

d ={'sentiment': y, 'document': preprocessed_corpus}
df = pd.DataFrame(data=d)
df.to_csv('preprocessed_1k.csv', index = False, header = False)
print(df)