import pandas as pd
import spacy
import csv
import itertools
import time

corpus = []
y = []
N = 1000

nlp = spacy.load('en_core_web_sm') # Language Model
#
# df = pd.read_csv('C:/Users/D072828/PycharmProjects/Thesis/venv/train_full.csv', header=None, nrows=100000)
# ds = df.sample(frac=1)
# print(ds)
# ds.to_csv('C:/Users/D072828/PycharmProjects/Thesis/venv/to_be_preprocessed_100_k.csv', index = False, header = False)
start = time.time()
with open('shuffled_yahoo_1k_test.csv', newline='', encoding="utf8") as csvfile:
    yelp = csv.reader(csvfile, delimiter=',')
    for row in itertools.islice(yelp, N):
         # clean_row = row[1].strip().replace('"','').replace(';','')
         rows = row[1:]
         text_in_doc = ''
         for e in rows:
             text_in_doc+= e + ' '
         clean_row = text_in_doc.strip().replace('"','').replace(';','')
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
        if not token.is_punct and token.is_alpha and (token.tag_ == 'NN' or token.tag_ == 'NNS' or token.tag_ == 'NNP' or token.tag_ == 'NNPS'):
            # preprocessed_row.append(token.lemma_)
            preprocessed_row+= (token.lemma_ + ' ')
            # preprocessed_row+= (token.text + ' ')
            # print(preprocessed_row)
    preprocessed_corpus.append(preprocessed_row)
    # print(preprocessed_corpus)
    # print(doc)
    # preprocessed_corpus.append(doc)


# print(len(preprocessed_corpus) == len(y))

d ={'sentiment': y, 'document': preprocessed_corpus}
df = pd.DataFrame(data=d)
df.to_csv('preprocessed_yahoo_1k_test.csv', index = False, header = False)
print(df)
end = time.time()
print('Runtime:', end - start)
