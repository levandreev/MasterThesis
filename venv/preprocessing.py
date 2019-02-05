import pandas as pd
import spacy
import csv
import itertools
import time
import re

corpus = []
y = []
N = 1000

# nlp = spacy.load('en_core_web_sm') # Language Model small
nlp = spacy.load('en_core_web_md') # Language Model medium
start = time.time()
with open('shuffled_yahoo_1k_test.csv', newline='', encoding="utf8") as csvfile:
    yelp = csv.reader(csvfile, delimiter=',')
    for row in itertools.islice(yelp, N):
         rows = row[1:]
         text_in_doc = ''
         #Matches any alphanumeric character; this is equivalent to the \w
         regex = re.compile("[^a-zA-Z0-9_']")
         for e in rows:
             text_in_doc+= e + ' '
         clean_row_with_duplicate_ws = regex.sub(' ', text_in_doc).strip()
         clean_row = re.sub(' +', ' ', clean_row_with_duplicate_ws)
         target_class = row[0]
         corpus.append(clean_row)
         y.append(int(target_class))

preprocessed_corpus = []
for row in corpus:
    doc = nlp(row)
    preprocessed_row = ""
    for token in doc:
        # token.is_oov = token out of vocabulary
        if not token.is_punct and token.is_alpha and (token.tag_ == 'NN' or token.tag_ == 'NNS' or token.tag_ == 'NNP' or token.tag_ == 'NNPS') and not token.is_oov:
            preprocessed_row += (token.lemma_ + ' ')
    preprocessed_corpus.append(preprocessed_row)

d = {'sentiment': y, 'document': preprocessed_corpus}
df = pd.DataFrame(data=d)
df.to_csv('preprocessed_yahoo_1k_test.csv', index = False, header = False)
print(df)
end = time.time()
print('Runtime:', end - start)
