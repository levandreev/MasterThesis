import numpy as np
import sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import time
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
import itertools



# classifier = GaussianNB()
# classifier = SVC(kernel='rbf') # Gives warning
classifier = LinearSVC(random_state=0, tol=1e-5)
# classifier = KNeighborsClassifier(n_neighbors=3)
corpus = []
y = []
N = 100

with open('C:/Users/D072828/PycharmProjects/Thesis/venv/preprocessed_10k.csv', newline='') as csvfile:
    yelp = csv.reader(csvfile, delimiter=',')
    for row in itertools.islice(yelp, N):
         clean_row = row[1].strip().replace('"','').replace(';','')
         # clean_row = row[0].strip().replace('"','').replace(';','')
         target_class = row[0]
         # target_class = clean_row[0]
         corpus.append(clean_row)
         # corpus.append(clean_row[2:])
         y.append(int(target_class))

tfidf_vectorizer = TfidfVectorizer(norm='l2')
matrix = tfidf_vectorizer.fit_transform(corpus)
sparse = tfidf_vectorizer.fit_transform(corpus).A
print(pd.DataFrame(matrix.todense(),columns=tfidf_vectorizer.get_feature_names()))

##Cross validation
print("CROSS VALIDATION:")
splits = 10
kf_total = StratifiedKFold(n_splits=splits, shuffle=True)
kf = KFold(n_splits=10 )
y1 = np.array(y)
avg_f1 = 0
avg_precision = 0
avg_recall = 0
plot_f1 =[]

start = time.time()
for train_index, test_index in kf_total.split(sparse, y1):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = sparse[train_index], sparse[test_index]
    y_train, y_test = y1[train_index], y1[test_index]
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro') # y_test = true y, y_pred = predicted y with the classifier
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    print('F1 score: ', f1)
    print('Precision score: ', precision)
    print('Recall score: ', recall)
    plot_f1.append(f1)
    avg_f1+= f1
    avg_precision += precision
    avg_recall += recall
end = time.time()
avg_recall = avg_recall/splits
avg_precision = avg_precision/splits
avg_f1 = avg_f1/splits

print('----------------------')
print('Average Precision:', avg_precision)
print('Average Recall:', avg_recall)
print('Average F1', avg_f1)
print('Runtime:', end - start)

print(plot_f1)
plt.plot(plot_f1, 'ro')
plt.show()

